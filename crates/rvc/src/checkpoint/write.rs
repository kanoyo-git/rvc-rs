use std::path::Path;
use std::{fs, io::Write};

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value as Json;
use tch::{Device, Kind, Tensor};
use tempfile::NamedTempFile;
use zip::{ZipWriter, write::SimpleFileOptions};

/// The safe subset of PyTorch's checkpoint representation used by RVC.
pub enum Value {
    Meta(Json),
    BFloat16,
    Tensor(Tensor),
    Dict(Vec<(Key, Value)>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
}

pub enum Key {
    String(String),
    Integer(i64),
}

impl From<&str> for Key {
    fn from(text: &str) -> Self {
        Self::String(text.to_owned())
    }
}

impl From<Json> for Value {
    fn from(value: Json) -> Self {
        Self::Meta(value)
    }
}

impl Value {
    pub fn tensors(named: impl IntoIterator<Item = (String, Tensor)>) -> Self {
        Self::Dict(
            named
                .into_iter()
                .map(|(name, tensor)| (Key::String(name), Self::Tensor(tensor)))
                .collect(),
        )
    }
}

pub fn write(path: &Path, value: &Value, overwrite: bool) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut staging = NamedTempFile::new_in(parent)?;
    let mut pickle = Pickle {
        bytes: vec![0x80, 0x02],
        tensors: Vec::new(),
    };
    pickle.value(value)?;
    pickle.bytes.push(b'.');
    let mut archive = ZipWriter::new(staging.as_file_mut());
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    archive.start_file("archive/data.pkl", opts)?;
    archive.write_all(&pickle.bytes)?;
    archive.start_file("archive/byteorder", opts)?;
    archive.write_all(b"little")?;
    for (index, tensor) in pickle.tensors.into_iter().enumerate() {
        let tensor = tensor
            .f_detach()?
            .f_to_device(Device::Cpu)?
            .f_contiguous()?;
        let bytes = tensor
            .numel()
            .checked_mul(tensor.kind().elt_size_in_bytes())
            .context("tensor storage overflow")?;
        let mut data = vec![0; bytes];
        tensor.f_copy_data_u8(&mut data, tensor.numel())?;
        archive.start_file(format!("archive/data/{index}"), opts)?;
        archive.write_all(&data)?;
    }
    archive.start_file("archive/version", opts)?;
    archive.write_all(b"3\n")?;
    archive.finish()?;
    if overwrite {
        staging.persist(path)?;
    } else {
        staging.persist_noclobber(path)?;
    }
    Ok(())
}

struct Pickle<'a> {
    bytes: Vec<u8>,
    tensors: Vec<&'a Tensor>,
}

impl<'a> Pickle<'a> {
    fn value(&mut self, value: &'a Value) -> Result<()> {
        match value {
            Value::Meta(json) => self.json(json)?,
            Value::BFloat16 => self.bytes.extend_from_slice(b"ctorch\nbfloat16\n"),
            Value::Tensor(tensor) => self.tensor(tensor)?,
            Value::Dict(fields) => {
                self.bytes.push(b'}');
                if !fields.is_empty() {
                    self.bytes.push(b'(');
                    for (key, entry) in fields {
                        match key {
                            Key::Integer(key) => self.integer(*key),
                            Key::String(key) => self.string(key)?,
                        };
                        self.value(entry)?;
                    }
                    self.bytes.push(b'u');
                }
            }
            Value::List(items) => {
                self.bytes.push(b']');
                if !items.is_empty() {
                    self.bytes.push(b'(');
                    for item in items {
                        self.value(item)?;
                    }
                    self.bytes.push(b'e');
                }
            }
            Value::Tuple(items) => {
                self.bytes.push(b'(');
                for item in items {
                    self.value(item)?;
                }
                self.bytes.push(b't');
            }
        }
        Ok(())
    }

    fn json(&mut self, value: &Json) -> Result<()> {
        match value {
            Json::Null => self.bytes.push(b'N'),
            Json::Bool(true) => self.bytes.push(0x88),
            Json::Bool(false) => self.bytes.push(0x89),
            Json::String(value) => self.string(value)?,
            Json::Number(value) => {
                if let Some(integer) = value.as_i64() {
                    self.integer(integer);
                } else {
                    self.bytes.push(b'G');
                    self.bytes.extend_from_slice(
                        &value
                            .as_f64()
                            .context("invalid checkpoint number")?
                            .to_be_bytes(),
                    );
                }
            }
            Json::Array(values) => {
                self.bytes.push(b']');
                if !values.is_empty() {
                    self.bytes.push(b'(');
                    for value in values {
                        self.json(value)?;
                    }
                    self.bytes.push(b'e');
                }
            }
            Json::Object(fields) => {
                self.bytes.push(b'}');
                if !fields.is_empty() {
                    self.bytes.push(b'(');
                    for (key, value) in fields {
                        self.string(key)?;
                        self.json(value)?;
                    }
                    self.bytes.push(b'u');
                }
            }
        }
        Ok(())
    }

    fn tensor(&mut self, tensor: &'a Tensor) -> Result<()> {
        let shape = tensor.size();
        let kind = match tensor.kind() {
            Kind::Half => "HalfStorage",
            Kind::Float => "FloatStorage",
            Kind::BFloat16 => "BFloat16Storage",
            Kind::Double => "DoubleStorage",
            Kind::Int64 => "LongStorage",
            Kind::Int => "IntStorage",
            Kind::Int16 => "ShortStorage",
            Kind::Int8 => "CharStorage",
            Kind::Uint8 => "ByteStorage",
            Kind::Bool => "BoolStorage",
            kind => bail!("unsupported checkpoint tensor type {kind:?}"),
        };
        ensure!(shape.len() <= 8, "tensor rank exceeds 8");
        let key = self.tensors.len().to_string();
        self.tensors.push(tensor);
        self.bytes
            .extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((");
        self.string("storage")?;
        self.bytes.extend_from_slice(b"ctorch\n");
        self.bytes.extend_from_slice(kind.as_bytes());
        self.bytes.push(b'\n');
        self.string(&key)?;
        self.string("cpu")?;
        self.integer(tensor.numel().try_into()?);
        self.bytes.extend_from_slice(b"tQ");
        self.integer(0);
        self.integers(shape.iter().copied());
        let mut stride = vec![0; shape.len()];
        let mut next = 1;
        for (stride, length) in stride.iter_mut().rev().zip(shape.iter().rev()) {
            *stride = next;
            next *= length.max(&1);
        }
        self.integers(stride);
        self.bytes.push(0x89);
        self.bytes
            .extend_from_slice(b"ccollections\nOrderedDict\n)RtR");
        Ok(())
    }

    fn integers(&mut self, ints: impl IntoIterator<Item = i64>) {
        self.bytes.push(b'(');
        for value in ints {
            self.integer(value);
        }
        self.bytes.push(b't');
    }

    fn integer(&mut self, value: i64) {
        if let Ok(value) = u8::try_from(value) {
            self.bytes.extend_from_slice(&[b'K', value]);
        } else if let Ok(value) = i32::try_from(value) {
            self.bytes.push(b'J');
            self.bytes.extend_from_slice(&value.to_le_bytes());
        } else {
            self.bytes.extend_from_slice(&[0x8a, 8]);
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn string(&mut self, value: &str) -> Result<()> {
        let len = u32::try_from(value.len())?;
        self.bytes.push(b'X');
        self.bytes.extend_from_slice(&len.to_le_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }
}
