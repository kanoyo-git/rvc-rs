use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value as Json};
use tch::{Device, Tensor};

use super::{
    Archive, Checkpoint, Id, MAX_PICKLE_BYTES, MAX_STORAGE_BYTES, Pickle, Storage, Value,
    demucs_kind, storage_bytes, string,
};

// Torch's binary stream serialization predates the default ZIP container.
// It stores five consecutive pickles, followed by length-prefixed storages.
pub(super) fn read(mut file: File) -> Result<Checkpoint> {
    let size = file.metadata()?.len();
    let mut header = Vec::new();
    file.by_ref()
        .take(MAX_PICKLE_BYTES)
        .read_to_end(&mut header)?;
    let tail = magic(&header)?;
    let (protocol, root, tail) = Pickle::parse_next(tail)?;
    ensure!(
        matches!(protocol[root], Value::Int(1001)),
        "unsupported legacy Torch version"
    );
    let (info, root, tail) = Pickle::parse_next(tail)?;
    let Value::Dict(fields) = &info[root] else {
        bail!("invalid legacy Torch system metadata");
    };
    ensure!(
        fields.iter().any(|&(key, val)| {
            matches!(&info[key], Value::String(key) if key == "little_endian")
                && matches!(info[val], Value::Bool(true))
        }),
        "unsupported legacy Torch byte order"
    );
    let (values, root, tail) = Pickle::parse_next(tail)?;
    let (keys, sequence, tail) = Pickle::parse_next(tail)?;
    let Value::List(keys_ordered) = &keys[sequence] else {
        bail!("missing legacy Torch storage keys");
    };
    let mut storages: HashMap<&str, &Storage> = HashMap::new();
    for value in &values {
        if let Value::Storage(storage) = value
            && let Some(previous) = storages.insert(&storage.key, storage)
        {
            ensure!(
                std::mem::discriminant(&previous.kind) == std::mem::discriminant(&storage.kind)
                    && previous.len == storage.len,
                "inconsistent legacy Torch storage descriptor"
            );
        }
    }
    let mut offset = (header.len() - tail.len()) as u64;
    let mut offsets = HashMap::new();
    let mut encoded_size = [0; 8];
    for &id in keys_ordered {
        let key = string(&keys[id])?;
        let storage = storages
            .remove(key)
            .context("unknown or duplicate legacy Torch storage")?;
        let bytes = storage_bytes(storage)?;
        ensure!(
            bytes <= MAX_STORAGE_BYTES,
            "legacy Torch storage exceeds import limits"
        );
        ensure!(
            offset.checked_add(8).is_some_and(|end| end <= size),
            "truncated legacy Torch storage header"
        );
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut encoded_size)?;
        ensure!(
            i64::from_le_bytes(encoded_size) == storage.len,
            "legacy Torch storage size mismatch"
        );
        offset += 8;
        offsets.insert(key.to_owned(), offset);
        offset = offset
            .checked_add(bytes)
            .context("legacy Torch storage offset overflow")?;
        ensure!(offset <= size, "truncated legacy Torch storage");
    }
    ensure!(
        storages.is_empty(),
        "legacy Torch stream is missing tensor storage"
    );
    Ok(Checkpoint {
        archive: Archive::Legacy { file, offsets },
        values,
        root,
    })
}

fn magic(bytes: &[u8]) -> Result<&[u8]> {
    let [0x80, protocol @ 2..=5, remaining @ ..] = bytes else {
        bail!("expected a binary legacy Torch stream");
    };
    let remaining = if *protocol >= 4 && remaining.first() == Some(&0x95) {
        let (_, tail) = remaining
            .split_at_checked(9)
            .context("truncated Torch magic frame")?;
        tail
    } else {
        remaining
    };
    const MAGIC: &[u8] = b"\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19.";
    remaining
        .strip_prefix(MAGIC)
        .context("invalid legacy Torch magic")
}

pub struct DemucsPackage {
    pub kind: &'static str,
    pub args: Vec<Json>,
    pub kwargs: Map<String, Json>,
}

impl Checkpoint {
    pub fn demucs_package(&self) -> Result<Option<DemucsPackage>> {
        let Some((class, arguments, keywords, _)) = self.demucs_fields()? else {
            return Ok(None);
        };
        let Value::Global(class) = &self.values[class] else {
            bail!("invalid Demucs model descriptor");
        };
        let kind = demucs_kind(class).context("unsupported Demucs model descriptor")?;
        let Json::Array(args) = self.json(arguments, 0, &mut 16_384)? else {
            bail!("invalid Demucs positional arguments");
        };
        let Json::Object(kwargs) = self.json(keywords, 0, &mut 16_384)? else {
            bail!("invalid Demucs keyword arguments");
        };
        Ok(Some(DemucsPackage { kind, args, kwargs }))
    }

    pub fn demucs_weights(&mut self, device: Device) -> Result<HashMap<String, Tensor>> {
        let state = self
            .demucs_fields()?
            .map(|(_, _, _, weights)| weights)
            .or_else(|| self.field("state_dict"))
            .or_else(|| self.field("model_state_dict"))
            .or_else(|| self.field("state"))
            .or_else(|| self.field("model"))
            .unwrap_or(self.root);
        ensure!(
            !self.dictionary(state)?.contains_key("__quantized"),
            "DiffQ quantized Demucs weights are unsupported"
        );
        self.weights_at(state, device, 4 * 1024 * 1024 * 1024)
    }

    fn demucs_fields(&self) -> Result<Option<(Id, Id, Id, Id)>> {
        if let Value::Tuple(fields) = &self.values[self.root] {
            let &[class, args, kwargs, state, ..] = fields.as_slice() else {
                bail!("invalid Demucs package");
            };
            return Ok(Some((class, args, kwargs, state)));
        }
        let Some(class) = self.field("klass") else {
            return Ok(None);
        };
        let args = self.field("args").context("Demucs package has no args")?;
        let kwargs = self
            .field("kwargs")
            .context("Demucs package has no kwargs")?;
        let state = self
            .field("state")
            .context("Demucs package has no weights")?;
        Ok(Some((class, args, kwargs, state)))
    }
}
