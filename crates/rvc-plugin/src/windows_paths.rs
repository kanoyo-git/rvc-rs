use std::env;
use std::ffi::{OsStr, OsString, c_char};
use std::fs;
use std::io::{Error, ErrorKind};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Mutex;

use anyhow::{Context, Result, ensure};

use crate::daw::ModelPaths;

const PROFILE_SECTION_LIMIT: usize = 32_768;
const UTF16_LE_BOM: &[u8] = &[0xff, 0xfe];
static SAVE: Mutex<()> = Mutex::new(());

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetPrivateProfileStringW(
        section: *const u16,
        key: *const u16,
        default: *const u16,
        output: *mut u16,
        capacity: u32,
        filename: *const u16,
    ) -> u32;
    fn GetPrivateProfileSectionW(
        section: *const u16,
        output: *mut u16,
        capacity: u32,
        filename: *const u16,
    ) -> u32;
    fn WritePrivateProfileSectionW(
        section: *const u16,
        values: *const u16,
        filename: *const u16,
    ) -> i32;
    fn MultiByteToWideChar(
        code_page: u32,
        flags: u32,
        input: *const c_char,
        input_size: i32,
        output: *mut u16,
        output_size: i32,
    ) -> i32;
}

fn profile_path() -> Result<Option<PathBuf>> {
    env::var_os("LOCALAPPDATA")
        .filter(|directory| !directory.is_empty())
        .map(|directory| {
            std::path::absolute(PathBuf::from(directory).join("RVCRealtime/settings.ini"))
        })
        .transpose()
        .context("resolve RVC Realtime settings.ini")
}

fn wide(value: impl AsRef<OsStr>) -> Vec<u16> {
    value.as_ref().encode_wide().chain([0]).collect()
}

fn read_key(filename: &[u16], section: &str, key: &str) -> PathBuf {
    let section = wide(section);
    let key = wide(key);
    let mut output = vec![0; PROFILE_SECTION_LIMIT];
    // SAFETY: all inputs are terminated, and the output owns capacity elements.
    let length = unsafe {
        GetPrivateProfileStringW(
            section.as_ptr(),
            key.as_ptr(),
            [0u16].as_ptr(),
            output.as_mut_ptr(),
            output.len() as u32,
            filename.as_ptr(),
        )
    } as usize;
    PathBuf::from(OsString::from_wide(&output[..length]))
}

pub(crate) fn read() -> ModelPaths {
    let Ok(Some(settings)) = profile_path() else {
        return ModelPaths::default();
    };
    if !settings.is_file() {
        return ModelPaths::default();
    }
    let filename = wide(settings);
    ModelPaths {
        model: read_key(&filename, "Paths", "ModelPath"),
        index: read_key(&filename, "Paths", "IndexPath"),
        rvc_root: read_key(&filename, "Paths", "RvcRoot"),
        legacy_python: read_key(&filename, "Paths", "PythonPath"),
    }
}

#[cfg(not(test))]
pub(crate) fn save(paths: &ModelPaths) -> Result<()> {
    if let Some(settings) = profile_path()? {
        save_to(&settings, paths)?;
    }
    Ok(())
}

fn ensure_unicode(settings: &Path) -> Result<()> {
    let bytes = match fs::read(settings) {
        Ok(bytes) if bytes.starts_with(UTF16_LE_BOM) => return Ok(()),
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            let mut unicode = UTF16_LE_BOM.to_vec();
            unicode.extend("[Paths]\r\n".encode_utf16().flat_map(u16::to_le_bytes));
            return rvc::audio::write_file(settings, &unicode, false);
        }
        Err(error) => return Err(error.into()),
    };
    let mut unicode = UTF16_LE_BOM.to_vec();
    if let Some(utf8) = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        unicode.extend(
            std::str::from_utf8(utf8)?
                .encode_utf16()
                .flat_map(u16::to_le_bytes),
        );
    } else if !bytes.is_empty() {
        let size = i32::try_from(bytes.len())?;
        let input = bytes.as_ptr().cast();
        // SAFETY: Windows reads exactly size bytes and reports the number of
        // UTF-16 units required for a conversion from the system code page.
        let needed = unsafe { MultiByteToWideChar(0, 0, input, size, ptr::null_mut(), 0) };
        if needed <= 0 {
            return Err(Error::last_os_error()).context("decode existing settings.ini");
        }
        let mut decoded = vec![0; needed as usize];
        // SAFETY: the sized input is unchanged and decoded has needed units.
        let converted =
            unsafe { MultiByteToWideChar(0, 0, input, size, decoded.as_mut_ptr(), needed) };
        if converted != needed {
            return Err(Error::last_os_error()).context("decode existing settings.ini");
        }
        unicode.extend(decoded.into_iter().flat_map(u16::to_le_bytes));
    }
    // Promote legacy ANSI files before the profile API writes new paths. The
    // whole original file is preserved when its text is encoded as UTF-16.
    rvc::audio::write_file(settings, &unicode, true)
}

fn save_to(settings: &Path, paths: &ModelPaths) -> Result<()> {
    let _guard = SAVE.lock().unwrap_or_else(|error| error.into_inner());
    let updates = [
        ("RvcRoot", &paths.rvc_root),
        ("ModelPath", &paths.model),
        ("IndexPath", &paths.index),
    ];
    let records: Vec<_> = updates
        .into_iter()
        .map(|(key, path)| {
            let mut record: Vec<u16> = key.encode_utf16().chain([u16::from(b'=')]).collect();
            for word in path.as_os_str().encode_wide() {
                ensure!(
                    word != 0 && word != 10 && word != 13,
                    "invalid settings path"
                );
                record.push(word);
            }
            record.push(0);
            Ok((key, record))
        })
        .collect::<Result<_>>()?;
    ensure_unicode(settings)?;
    let filename = wide(settings);
    let section = wide("Paths");
    let mut old = vec![0; PROFILE_SECTION_LIMIT];
    // SAFETY: section and filename are terminated and old owns the output buffer.
    let length = unsafe {
        GetPrivateProfileSectionW(
            section.as_ptr(),
            old.as_mut_ptr(),
            old.len() as u32,
            filename.as_ptr(),
        )
    } as usize;
    ensure!(
        length < old.len() - 2,
        "settings.ini Paths section is too long"
    );
    let mut values = Vec::new();
    let mut seen = [false; 3];
    for record in old[..length]
        .split(|&word| word == 0)
        .filter(|record| !record.is_empty())
    {
        let key_end = record.iter().position(|&word| word == u16::from(b'='));
        let replacement = key_end.and_then(|end| {
            let key = String::from_utf16_lossy(&record[..end]);
            records
                .iter()
                .position(|(name, _)| name.eq_ignore_ascii_case(key.trim()))
        });
        if let Some(position) = replacement {
            if !seen[position] {
                values.extend_from_slice(&records[position].1);
                seen[position] = true;
            }
        } else {
            values.extend(record);
            values.push(0);
        }
    }
    for (written, (_, record)) in seen.into_iter().zip(records) {
        if !written {
            values.extend(record);
        }
    }
    values.push(0);
    ensure!(
        values.len() * 2 <= u16::MAX as usize,
        "settings.ini Paths section is too long"
    );
    // SAFETY: values ends with a double NUL. Windows replaces this section
    // atomically, retaining existing keys supplied in the merged buffer.
    let written = unsafe {
        WritePrivateProfileSectionW(section.as_ptr(), values.as_ptr(), filename.as_ptr())
    };
    if written == 0 {
        return Err(Error::last_os_error()).context("write settings.ini Paths section");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_only_the_native_fields_of_legacy_profile() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let scratch = env::temp_dir().join(format!(
            "rvc-paths-{}-{}",
            std::process::id(),
            stamp.as_nanos()
        ));
        fs::create_dir(&scratch).unwrap();
        let file = scratch.join("settings.ini");
        fs::write(&file, b"[Paths]\r\npythonpath=C:\\python.exe\r\nModelPath=old.pth\r\nIndexPath=old.index\r\nExtra=keep\r\n[Other]\r\nSaved=yes\r\n").unwrap();
        let paths = ModelPaths {
            model: PathBuf::from("C:\\モデル\\русский.pth"),
            rvc_root: PathBuf::from("C:\\RVC"),
            legacy_python: PathBuf::from("unused"),
            ..ModelPaths::default()
        };
        save_to(&file, &paths).unwrap();
        let filename = wide(&file);
        assert_eq!(read_key(&filename, "Paths", "ModelPath"), paths.model);
        assert_eq!(read_key(&filename, "Paths", "RvcRoot"), paths.rvc_root);
        assert_eq!(read_key(&filename, "Paths", "IndexPath"), PathBuf::new());
        assert_eq!(
            read_key(&filename, "Paths", "PythonPath"),
            PathBuf::from("C:\\python.exe")
        );
        assert_eq!(read_key(&filename, "Paths", "Extra"), PathBuf::from("keep"));
        assert_eq!(read_key(&filename, "Other", "Saved"), PathBuf::from("yes"));

        let fresh = scratch.join("new/settings.ini");
        save_to(&fresh, &paths).unwrap();
        assert_eq!(read_key(&wide(&fresh), "Paths", "ModelPath"), paths.model);
        fs::remove_file(&fresh).unwrap();
        fs::remove_dir(fresh.parent().unwrap()).unwrap();
        fs::remove_file(file).unwrap();
        fs::remove_dir(scratch).unwrap();
    }
}
