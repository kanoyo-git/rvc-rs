//! PyTorch ZIP checkpoints. Pickle opcodes are interpreted as data, with an
//! allowlist for tensor references and OrderedDict.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value as Json;
use zip::ZipArchive;

#[cfg(feature = "inference")]
mod legacy;
#[cfg(feature = "inference")]
pub(crate) mod optimizer;
#[cfg(feature = "inference")]
pub mod write;

const MAX_PICKLE_BYTES: u64 = 16 * 1024 * 1024;
#[cfg(feature = "inference")]
const MAX_STORAGE_BYTES: u64 = 512 * 1024 * 1024;

type Id = usize;

#[derive(Clone, Debug)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    NumpyDtype { code: String, endian: char },
    Global(String),
    List(Vec<Id>),
    Tuple(Vec<Id>),
    Dict(Vec<(Id, Id)>),
    Storage(Storage),
    Tensor(TensorRef),
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Kind is used by inference builds.
enum Kind {
    Half,
    BFloat16,
    Float,
    Double,
    Long,
    Int,
    Short,
    Char,
    Byte,
    Bool,
}

impl Kind {
    fn parse(class: &str) -> Result<Self> {
        Ok(match class.strip_prefix("torch.").unwrap_or_default() {
            "HalfStorage" => Self::Half,
            "BFloat16Storage" => Self::BFloat16,
            "FloatStorage" => Self::Float,
            "DoubleStorage" => Self::Double,
            "LongStorage" => Self::Long,
            "IntStorage" => Self::Int,
            "ShortStorage" => Self::Short,
            "CharStorage" => Self::Char,
            "ByteStorage" => Self::Byte,
            "BoolStorage" => Self::Bool,
            _ => bail!("unsupported PyTorch storage class {class}"),
        })
    }

    #[cfg(feature = "inference")]
    fn tensor_kind(self) -> tch::Kind {
        match self {
            Self::Half => tch::Kind::Half,
            Self::BFloat16 => tch::Kind::BFloat16,
            Self::Float => tch::Kind::Float,
            Self::Double => tch::Kind::Double,
            Self::Long => tch::Kind::Int64,
            Self::Int => tch::Kind::Int,
            Self::Short => tch::Kind::Int16,
            Self::Char => tch::Kind::Int8,
            Self::Byte => tch::Kind::Uint8,
            Self::Bool => tch::Kind::Bool,
        }
    }
}

#[derive(Clone, Debug)]
struct Storage {
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    key: String,
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    kind: Kind,
    len: i64,
}

#[derive(Clone, Debug)]
struct TensorRef {
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    storage: Storage,
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    offset: i64,
    shape: Vec<i64>,
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    stride: Vec<i64>,
}

pub struct Checkpoint {
    #[cfg_attr(not(feature = "inference"), allow(dead_code))]
    archive: Archive,
    values: Vec<Value>,
    root: Id,
}

#[cfg_attr(not(feature = "inference"), allow(dead_code))]
enum Archive {
    Zip {
        file: Box<ZipArchive<File>>,
        prefix: String,
    },
    #[cfg(feature = "inference")]
    Legacy {
        file: File,
        offsets: HashMap<String, u64>,
    },
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self> {
        Self::read(path).with_context(|| format!("reading checkpoint {}", path.display()))
    }

    fn read(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        #[cfg(feature = "inference")]
        let file = {
            use std::io::{Seek, SeekFrom};
            let mut file = file;
            let mut magic = [0; 4];
            file.read_exact(&mut magic)?;
            file.seek(SeekFrom::Start(0))?;
            if magic[0] == 0x80 {
                return legacy::read(file);
            }
            file
        };
        let mut archive = ZipArchive::new(file)?;
        ensure!(
            archive.len() <= 100_000,
            "too many entries in checkpoint archive"
        );
        let names: Vec<_> = archive
            .file_names()
            .filter(|name| name.ends_with("/data.pkl") || *name == "data.pkl")
            .collect();
        ensure!(
            names.len() == 1,
            "expected exactly one PyTorch data.pkl entry"
        );
        let pickle_name = names[0].to_owned();
        let prefix = pickle_name.trim_end_matches("data.pkl").to_owned();
        let mut bytes = Vec::new();
        archive
            .by_name(&pickle_name)?
            .take(MAX_PICKLE_BYTES + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= MAX_PICKLE_BYTES,
            "checkpoint metadata exceeds 16 MiB"
        );
        if let Ok(byteorder) = archive.by_name(&format!("{prefix}byteorder")) {
            let mut order = String::new();
            byteorder.take(12).read_to_string(&mut order)?;
            ensure!(
                order == "little",
                "unsupported checkpoint byte order {order:?}"
            );
        }
        let (values, root) = Pickle::parse(&bytes)?;
        let archive = Archive::Zip {
            file: Box::new(archive),
            prefix,
        };
        Ok(Self {
            archive,
            values,
            root,
        })
    }

    fn dictionary(&self, id: Id) -> Result<BTreeMap<&str, Id>> {
        let Value::Dict(entries) = &self.values[id] else {
            bail!("expected a checkpoint dictionary");
        };
        entries
            .iter()
            .map(|&(key, value)| Ok((string(&self.values[key])?, value)))
            .collect()
    }

    fn field(&self, name: &str) -> Option<Id> {
        if let Value::Dict(entries) = &self.values[self.root] {
            entries.iter().rev().find_map(|&(key, value)| {
                matches!(&self.values[key], Value::String(key) if key == name).then_some(value)
            })
        } else {
            None
        }
    }

    pub fn metadata(&self, name: &str) -> Result<Option<Json>> {
        let mut budget = 16_384;
        self.field(name)
            .map(|id| self.json(id, 0, &mut budget))
            .transpose()
    }

    pub fn has(&self, name: &str) -> bool {
        self.field(name).is_some()
    }

    #[cfg(feature = "inference")]
    pub fn compact_metadata(&self) -> Result<BTreeMap<String, Json>> {
        self.dictionary(self.root)?
            .keys()
            .filter(|&&name| name != "weight")
            .map(|&name| {
                Ok((
                    name.to_owned(),
                    self.metadata(name)?.context("metadata field disappeared")?,
                ))
            })
            .collect()
    }

    fn json(&self, id: Id, depth: u8, remaining: &mut usize) -> Result<Json> {
        ensure!(
            depth < 16 && *remaining > 0,
            "checkpoint metadata is too deeply nested or too large"
        );
        *remaining -= 1;
        Ok(match &self.values[id] {
            Value::None => Json::Null,
            Value::String(s) => Json::String(s.clone()),
            Value::Bool(v) => Json::Bool(*v),
            Value::Int(v) => (*v).into(),
            Value::Float(v) => serde_json::Number::from_f64(*v)
                .context("non-finite checkpoint metadata")?
                .into(),
            Value::Tuple(items) | Value::List(items) => Json::Array(
                items
                    .iter()
                    .map(|&item| self.json(item, depth + 1, remaining))
                    .collect::<Result<_>>()?,
            ),
            Value::Dict(_) => Json::Object(
                self.dictionary(id)?
                    .into_iter()
                    .map(|(key, item)| Ok((key.to_owned(), self.json(item, depth + 1, remaining)?)))
                    .collect::<Result<_>>()?,
            ),
            _ => bail!("invalid data type in checkpoint metadata"),
        })
    }

    pub fn tensor_shape(&self, section: Option<&str>, name: &str) -> Result<&[i64]> {
        let dictionary = self.dictionary(self.section(section)?)?;
        let id = dictionary
            .get(name)
            .with_context(|| format!("checkpoint is missing {name}"))?;
        let Value::Tensor(tensor) = &self.values[*id] else {
            bail!("checkpoint entry {name} is not a tensor");
        };
        Ok(&tensor.shape)
    }

    fn section(&self, section: Option<&str>) -> Result<Id> {
        match section {
            None => Ok(self.root),
            Some(name) => self
                .field(name)
                .with_context(|| format!("checkpoint is missing section {name}")),
        }
    }

    #[cfg(feature = "inference")]
    pub fn weights(
        &mut self,
        section: Option<&str>,
        device: tch::Device,
    ) -> Result<HashMap<String, tch::Tensor>> {
        self.weights_at(self.section(section)?, device, 2 * 1024 * 1024 * 1024)
    }

    #[cfg(feature = "inference")]
    fn weights_at(
        &mut self,
        id: Id,
        device: tch::Device,
        budget: u64,
    ) -> Result<HashMap<String, tch::Tensor>> {
        let references = self
            .dictionary(id)?
            .into_iter()
            .map(|(name, id)| {
                let Value::Tensor(tensor) = &self.values[id] else {
                    bail!("checkpoint entry {name} is not a tensor");
                };
                Ok((name.to_owned(), tensor.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tensors = HashMap::with_capacity(references.len());
        let mut loaded_bytes = 0_u64;
        for (name, reference) in references {
            let len = storage_bytes(&reference.storage)?;
            loaded_bytes = loaded_bytes
                .checked_add(len)
                .context("checkpoint size overflow")?;
            ensure!(
                loaded_bytes <= budget,
                "tensor storage exceeds import limits"
            );
            let tensor = self
                .tensor(&reference, device)
                .with_context(|| format!("loading {name}"))?;
            let kind = tensor.kind();
            let tensor = if matches!(
                kind,
                tch::Kind::Half | tch::Kind::BFloat16 | tch::Kind::Double
            ) {
                tensor.f_to_kind(tch::Kind::Float)?
            } else {
                tensor
            };
            tensors.insert(name, tensor);
        }
        Ok(tensors)
    }

    #[cfg(feature = "inference")]
    pub fn tensor_names(&self, section: Option<&str>) -> Result<Vec<String>> {
        let Value::Dict(entries) = &self.values[self.section(section)?] else {
            bail!("expected a checkpoint dictionary");
        };
        entries
            .iter()
            .filter(|&&(_, id)| matches!(self.values[id], Value::Tensor(_)))
            .map(|&(key, _)| Ok(string(&self.values[key])?.to_owned()))
            .collect()
    }

    #[cfg(feature = "inference")]
    fn tensor(&mut self, reference: &TensorRef, device: tch::Device) -> Result<tch::Tensor> {
        let storage = &reference.storage;
        let len = storage_bytes(storage)?;
        let kind = storage.kind.tensor_kind();
        ensure!(
            len <= MAX_STORAGE_BYTES,
            "tensor storage exceeds import limits"
        );
        let count = reference
            .shape
            .iter()
            .try_fold(1u64, |size, &d| size.checked_mul(d as u64))
            .context("tensor shape overflow")?;
        ensure!(
            count <= MAX_STORAGE_BYTES / kind.elt_size_in_bytes() as u64,
            "materialized tensor exceeds import limits"
        );
        let mut data = Vec::with_capacity(len as usize);
        match &mut self.archive {
            Archive::Zip { file, prefix } => {
                let entry = file.by_name(&format!("{prefix}data/{}", storage.key))?;
                ensure!(entry.size() == len, "unexpected tensor storage size");
                entry.take(len + 1).read_to_end(&mut data)?;
            }
            Archive::Legacy { file, offsets } => {
                use std::io::{Seek, SeekFrom};
                let offset = offsets
                    .get(&storage.key)
                    .context("missing legacy storage offset")?;
                file.seek(SeekFrom::Start(*offset))?;
                file.take(len).read_to_end(&mut data)?;
            }
        }
        ensure!(data.len() as u64 == len, "incomplete tensor storage");
        Ok(tch::Tensor::f_from_data_size(&data, &[storage.len], kind)?
            .f_as_strided(&reference.shape, &reference.stride, reference.offset)?
            .f_contiguous()?
            .f_to_device(device)?)
    }
}

#[cfg(feature = "inference")]
fn storage_bytes(storage: &Storage) -> Result<u64> {
    (storage.len as u64)
        .checked_mul(storage.kind.tensor_kind().elt_size_in_bytes() as u64)
        .context("tensor storage size overflow")
}

fn string(value: &Value) -> Result<&str> {
    match value {
        Value::String(text) => Ok(text),
        _ => bail!("expected a string in pickle data"),
    }
}

fn nonnegative(value: &Value) -> Result<i64> {
    match value {
        Value::Int(n) if *n >= 0 => Ok(*n),
        _ => bail!("expected a non-negative integer in tensor reference"),
    }
}

struct Pickle<'a> {
    source: &'a [u8],
    values: Vec<Value>,
    stack: Vec<Id>,
    marks: Vec<usize>,
    memo: HashMap<u32, Id>,
}

impl<'a> Pickle<'a> {
    fn parse(source: &'a [u8]) -> Result<(Vec<Value>, Id)> {
        let (values, root, _) = Self::parse_next(source)?;
        Ok((values, root))
    }

    fn parse_next(source: &'a [u8]) -> Result<(Vec<Value>, Id, &'a [u8])> {
        let mut p = Self {
            source,
            values: Vec::new(),
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
        };
        loop {
            let opcode = p.byte()?;
            match opcode {
                b'.' => {
                    ensure!(
                        p.stack.len() == 1 && p.marks.is_empty(),
                        "invalid pickle stack at STOP"
                    );
                    return Ok((p.values, p.stack[0], p.source));
                }
                0x80 => {
                    let version = p.byte()?;
                    ensure!(
                        (2..=5).contains(&version),
                        "unsupported pickle protocol {version}"
                    );
                }
                0x95 => {
                    let size = u64::from_le_bytes(p.array()?);
                    ensure!(size <= p.source.len() as u64, "truncated pickle frame");
                }
                b'(' => p.marks.push(p.stack.len()),
                b'N' => p.push(Value::None),
                0x88 | 0x89 => p.push(Value::Bool(opcode == 0x88)),
                b'I' => {
                    let number = p.line()?.parse::<i64>()?;
                    p.push(Value::Int(number));
                }
                b'K' | b'M' | b'J' => {
                    let number = match opcode {
                        b'K' => i64::from(p.byte()?),
                        b'M' => i64::from(u16::from_le_bytes(p.array()?)),
                        _ => i64::from(i32::from_le_bytes(p.array()?)),
                    };
                    p.push(Value::Int(number));
                }
                0x8a | 0x8b => {
                    let count = if opcode == 0x8a {
                        usize::from(p.byte()?)
                    } else {
                        p.u32()? as usize
                    };
                    ensure!(count <= 8, "pickle integer exceeds 64 bits");
                    let bytes = p.take(count)?;
                    let mut padded = [if bytes.last().is_some_and(|b| b & 0x80 != 0) {
                        0xff
                    } else {
                        0
                    }; 8];
                    padded[..count].copy_from_slice(bytes);
                    p.push(Value::Int(i64::from_le_bytes(padded)));
                }
                b'G' => {
                    let number = f64::from_be_bytes(p.array()?);
                    p.push(Value::Float(number));
                }
                b'B' | b'C' | 0x8e => {
                    let count = match opcode {
                        b'B' => p.u32()? as usize,
                        b'C' => usize::from(p.byte()?),
                        _ => usize::try_from(u64::from_le_bytes(p.array()?))?,
                    };
                    let data = p.take(count)?.to_vec();
                    p.push(Value::Bytes(data));
                }
                b'X' | 0x8c | 0x8d => {
                    let count = match opcode {
                        b'X' => p.u32()? as usize,
                        0x8c => p.byte()? as usize,
                        _ => usize::try_from(u64::from_le_bytes(p.array()?))?,
                    };
                    let data = std::str::from_utf8(p.take(count)?)?.to_owned();
                    p.push(Value::String(data));
                }
                b'c' => {
                    let module = p.line()?;
                    let name = p.line()?;
                    p.global(module, name)?;
                }
                0x93 => {
                    let name = p.pop()?;
                    let module = p.pop()?;
                    let name = string(&p.values[name])?.to_owned();
                    let module = string(&p.values[module])?.to_owned();
                    p.global(&module, &name)?;
                }
                b'q' | b'r' | 0x94 => {
                    let index = match opcode {
                        b'q' => u32::from(p.byte()?),
                        b'r' => p.u32()?,
                        _ => p.memo.len() as u32,
                    };
                    p.memo.insert(index, p.top()?);
                }
                b'h' | b'j' => {
                    let index = if opcode == b'h' {
                        u32::from(p.byte()?)
                    } else {
                        p.u32()?
                    };
                    p.stack.push(
                        *p.memo
                            .get(&index)
                            .context("missing pickle memo reference")?,
                    );
                }
                b')' => p.push(Value::Tuple(Vec::new())),
                b']' => p.push(Value::List(Vec::new())),
                b'}' => p.push(Value::Dict(Vec::new())),
                b't' | b'l' | b'd' => {
                    let items = p.marked()?;
                    let value = match opcode {
                        b't' => Value::Tuple(items),
                        b'l' => Value::List(items),
                        _ => Value::Dict(pairs(&items)?),
                    };
                    p.push(value);
                }
                0x85..=0x87 => {
                    let count = usize::from(opcode - 0x84);
                    ensure!(
                        p.stack.len() >= p.marks.last().copied().unwrap_or(0) + count,
                        "pickle tuple underflow"
                    );
                    let items = p.stack.split_off(p.stack.len() - count);
                    p.push(Value::Tuple(items));
                }
                b'a' | b'e' => {
                    let items = if opcode == b'a' {
                        vec![p.pop()?]
                    } else {
                        p.marked()?
                    };
                    let list = p.top()?;
                    let Value::List(target) = &mut p.values[list] else {
                        bail!("pickle APPEND target is not a list");
                    };
                    target.extend(items);
                }
                b's' | b'u' => {
                    let items = if opcode == b's' {
                        let value = p.pop()?;
                        vec![p.pop()?, value]
                    } else {
                        p.marked()?
                    };
                    let target = p.top()?;
                    let Value::Dict(dict) = &mut p.values[target] else {
                        bail!("pickle SETITEM target is not a dictionary");
                    };
                    dict.extend(pairs(&items)?);
                }
                b'Q' => p.persistent()?,
                b'R' => p.reduce()?,
                0x81 => p.new_object()?,
                b'b' => {
                    let attributes = p.pop()?;
                    let target = p.top()?;
                    p.build(target, attributes)?;
                }
                _ => bail!("unsupported pickle opcode 0x{opcode:02x}"),
            }
            ensure!(
                p.values.len() <= 500_000 && p.stack.len() <= 500_000,
                "checkpoint metadata exceeds object limits"
            );
        }
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into()?)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let (head, tail) = self
            .source
            .split_at_checked(count)
            .context("truncated pickle")?;
        self.source = tail;
        Ok(head)
    }

    fn line(&mut self) -> Result<&'a str> {
        let end = self
            .source
            .iter()
            .take(256)
            .position(|&b| b == b'\n')
            .context("malformed pickle line")?;
        let bytes = self.take(end + 1)?;
        Ok(std::str::from_utf8(&bytes[..end])?)
    }

    fn push(&mut self, value: Value) {
        self.stack.push(self.values.len());
        self.values.push(value);
    }

    fn top(&self) -> Result<Id> {
        ensure!(
            self.stack.len() > self.marks.last().copied().unwrap_or(0),
            "pickle stack underflow"
        );
        self.stack.last().copied().context("pickle stack underflow")
    }

    fn pop(&mut self) -> Result<Id> {
        let value = self.top()?;
        self.stack.pop();
        Ok(value)
    }

    fn marked(&mut self) -> Result<Vec<Id>> {
        let start = self.marks.pop().context("missing pickle mark")?;
        Ok(self.stack.split_off(start))
    }

    fn tuple(&self, id: Id) -> Result<&[Id]> {
        match &self.values[id] {
            Value::Tuple(items) => Ok(items),
            _ => bail!("expected a pickle tuple"),
        }
    }

    fn global(&mut self, module: &str, name: &str) -> Result<()> {
        let class = format!("{module}.{name}");
        ensure!(
            matches!(
                class.as_str(),
                "collections.OrderedDict"
                    | "torch._utils._rebuild_tensor"
                    | "torch._utils._rebuild_tensor_v2"
                    | "fractions.Fraction"
                    | "numpy.dtype"
                    | "numpy.core.multiarray.scalar"
                    | "numpy._core.multiarray.scalar"
                    | "_codecs.encode"
                    | "torch.torch_version.TorchVersion"
            ) || demucs_kind(&class).is_some()
                || Kind::parse(&class).is_ok(),
            "unsupported PyTorch pickle global {class}"
        );
        self.push(Value::Global(class));
        Ok(())
    }

    fn build(&mut self, target: Id, attributes: Id) -> Result<()> {
        match &self.values[target] {
            Value::Dict(_) => ensure!(
                matches!(self.values[attributes], Value::Dict(_)),
                "invalid ordered dictionary metadata"
            ),
            Value::NumpyDtype { code, .. } => {
                let [_, endian, ..] = self.tuple(attributes)? else {
                    bail!("invalid scalar dtype metadata");
                };
                let endian = match string(&self.values[*endian])? {
                    "<" | "|" => '<',
                    ">" => '>',
                    _ => bail!("invalid scalar dtype byte order"),
                };
                self.values[target] = Value::NumpyDtype {
                    code: code.clone(),
                    endian,
                };
            }
            _ => bail!("unsupported pickle object state"),
        }
        Ok(())
    }

    fn persistent(&mut self) -> Result<()> {
        let id = self.pop()?;
        let items = self.tuple(id)?;
        if let [name, class, _, _] = items
            && string(&self.values[*name])? == "module"
        {
            let Value::Global(class_name) = &self.values[*class] else {
                bail!("invalid legacy module descriptor");
            };
            ensure!(
                demucs_kind(class_name).is_some(),
                "unsupported pickle module descriptor"
            );
            self.stack.push(*class);
            return Ok(());
        }
        let [name, class, key, _, len, rest @ ..] = items else {
            bail!("invalid PyTorch storage reference");
        };
        ensure!(
            string(&self.values[*name])? == "storage",
            "invalid persistent reference"
        );
        ensure!(
            rest.is_empty() || (rest.len() == 1 && matches!(self.values[rest[0]], Value::None)),
            "unsupported storage view"
        );
        let Value::Global(class) = &self.values[*class] else {
            bail!("missing PyTorch storage class");
        };
        let kind = Kind::parse(class)?;
        let key = string(&self.values[*key])?.to_owned();
        ensure!(
            !key.is_empty() && key.len() < 32 && key.bytes().all(|byte| byte.is_ascii_digit()),
            "invalid PyTorch storage key"
        );
        let len = nonnegative(&self.values[*len])?;
        self.push(Value::Storage(Storage { key, kind, len }));
        Ok(())
    }

    fn new_object(&mut self) -> Result<()> {
        let args = self.pop()?;
        let class = self.pop()?;
        let Value::Global(name) = &self.values[class] else {
            bail!("invalid checkpoint string descriptor");
        };
        ensure!(
            name == "torch.torch_version.TorchVersion",
            "unsupported checkpoint object descriptor {name}"
        );
        let [value] = self.tuple(args)? else {
            bail!("invalid Torch version metadata");
        };
        string(&self.values[*value])?;
        self.stack.push(*value);
        Ok(())
    }

    fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let class = self.pop()?;
        let args = self.tuple(args)?;
        let Value::Global(class) = &self.values[class] else {
            bail!("pickle REDUCE does not reference a known descriptor");
        };
        if class == "collections.OrderedDict" && args.is_empty() {
            self.push(Value::Dict(Vec::new()));
        } else if class == "fractions.Fraction" {
            let (num, den) = match args {
                [num, den] => {
                    let (Value::Int(num), Value::Int(den)) =
                        (&self.values[*num], &self.values[*den])
                    else {
                        bail!("invalid fractional checkpoint metadata");
                    };
                    (*num, *den)
                }
                [value] => {
                    let (num, den) = string(&self.values[*value])?
                        .split_once('/')
                        .context("invalid fractional checkpoint metadata")?;
                    (num.parse()?, den.parse()?)
                }
                _ => bail!("invalid fractional checkpoint metadata"),
            };
            ensure!(den != 0, "invalid fractional checkpoint metadata");
            self.push(Value::Float(num as f64 / den as f64));
        } else if class == "_codecs.encode" {
            let [text, encoding] = args else {
                bail!("invalid byte string metadata");
            };
            ensure!(
                string(&self.values[*encoding])? == "latin1",
                "unsupported byte string encoding"
            );
            let bytes = string(&self.values[*text])?
                .chars()
                .map(|c| u8::try_from(u32::from(c)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            self.push(Value::Bytes(bytes));
        } else if class == "numpy.dtype" {
            let [dtype, ..] = args else {
                bail!("missing scalar dtype descriptor");
            };
            let code = string(&self.values[*dtype])?.to_owned();
            ensure!(
                matches!(code.as_str(), "f4" | "f8" | "i2" | "i4" | "i8" | "u1"),
                "unsupported numpy scalar dtype"
            );
            self.push(Value::NumpyDtype { code, endian: '<' });
        } else if matches!(
            class.as_str(),
            "numpy.core.multiarray.scalar" | "numpy._core.multiarray.scalar"
        ) {
            let [dtype, bytes] = args else {
                bail!("invalid numpy scalar metadata");
            };
            let Value::NumpyDtype { code, endian } = &self.values[*dtype] else {
                bail!("missing scalar dtype");
            };
            let Value::Bytes(bytes) = &self.values[*bytes] else {
                bail!("missing scalar bytes");
            };
            ensure!(bytes.len() <= 8, "invalid scalar length");
            let mut data = bytes.to_vec();
            if *endian == '>' {
                data.reverse();
            }
            let value = match code.as_str() {
                "f4" => Value::Float(f64::from(f32::from_le_bytes(data.as_slice().try_into()?))),
                "f8" => Value::Float(f64::from_le_bytes(data.as_slice().try_into()?)),
                "i2" => Value::Int(i64::from(i16::from_le_bytes(data.as_slice().try_into()?))),
                "i4" => Value::Int(i64::from(i32::from_le_bytes(data.as_slice().try_into()?))),
                "i8" => Value::Int(i64::from_le_bytes(data.as_slice().try_into()?)),
                "u1" if data.len() == 1 => Value::Int(i64::from(data[0])),
                _ => bail!("unsupported numpy scalar metadata"),
            };
            self.push(value);
        } else if matches!(
            class.as_str(),
            "torch._utils._rebuild_tensor" | "torch._utils._rebuild_tensor_v2"
        ) {
            let [storage, offset, shape, stride, ..] = args else {
                bail!("invalid PyTorch tensor reference");
            };
            let Value::Storage(storage) = &self.values[*storage] else {
                bail!("missing PyTorch tensor storage");
            };
            let offset = nonnegative(&self.values[*offset])?;
            let shape = self.dimensions(*shape)?;
            let stride = self.dimensions(*stride)?;
            ensure!(shape.len() == stride.len(), "invalid tensor stride");
            let end =
                shape
                    .iter()
                    .zip(&stride)
                    .try_fold(offset, |end, (&size, &step)| -> Result<_> {
                        end.checked_add(
                            size.saturating_sub(1)
                                .max(0)
                                .checked_mul(step)
                                .context("invalid tensor dimension")?,
                        )
                        .context("invalid tensor span")
                    })?;
            ensure!(
                if shape.contains(&0) {
                    offset <= storage.len
                } else {
                    end < storage.len
                },
                "tensor reference exceeds storage"
            );
            self.push(Value::Tensor(TensorRef {
                storage: storage.clone(),
                offset,
                shape,
                stride,
            }));
        } else {
            bail!("unsupported pickle REDUCE descriptor {class}");
        }
        Ok(())
    }

    fn dimensions(&self, id: Id) -> Result<Vec<i64>> {
        let items = self.tuple(id)?;
        ensure!(items.len() <= 8, "tensor rank exceeds 8");
        items
            .iter()
            .map(|&item| nonnegative(&self.values[item]))
            .collect()
    }
}

fn demucs_kind(class: &str) -> Option<&'static str> {
    match class {
        "demucs.model.Demucs" => Some("demucs"),
        "demucs.demucs.Demucs" => Some("demucs_v3"),
        "demucs.tasnet.ConvTasNet" => Some("conv_tasnet"),
        "demucs.hdemucs.HDemucs" => Some("hdemucs"),
        "demucs.htdemucs.HTDemucs" => Some("htdemucs"),
        _ => None,
    }
}

fn pairs(items: &[Id]) -> Result<Vec<(Id, Id)>> {
    let (pairs, remaining) = items.as_chunks::<2>();
    ensure!(remaining.is_empty(), "invalid pickle dictionary size");
    Ok(pairs.iter().map(|&[key, value]| (key, value)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_torch_version_as_plain_metadata() {
        let (values, root) = Pickle::parse(
            b"\x80\x02ctorch.torch_version\nTorchVersion\nX\x05\x00\x00\x002.7.0\x85\x81.",
        )
        .unwrap();
        assert_eq!(string(&values[root]).unwrap(), "2.7.0");
    }

    #[test]
    fn rejects_executable_pickle_globals() {
        let error = Pickle::parse(b"\x80\x02cposix\nsystem\n.").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported PyTorch pickle global posix.system")
        );
    }

    #[test]
    fn keeps_references_to_mutated_memo_entries() {
        let (values, root) =
            Pickle::parse(b"\x80\x02}q\x00X\x01\x00\x00\x00a]q\x01K\x03as.").unwrap();
        let Value::Dict(items) = &values[root] else {
            panic!("root is not a dictionary");
        };
        let Value::List(list) = &values[items[0].1] else {
            panic!("entry is not a list");
        };
        assert!(matches!(values[list[0]], Value::Int(3)));
    }
}
