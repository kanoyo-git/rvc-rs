//! The inference-only subset of ONNX's public protobuf schema.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use prost::Message;
use tch::{Device, Kind, Tensor};

use crate::model::weights::Weights;

const FLOAT: i32 = 1;
const FLOAT16: i32 = 10;
const INT64: i32 = 7;
const WEIGHTS: &str = "weights.bin";

#[derive(Message)]
struct Model {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(string, tag = "2")]
    producer: String,
    #[prost(message, optional, tag = "7")]
    graph: Option<Graph>,
    #[prost(message, repeated, tag = "8")]
    opsets: Vec<Opset>,
}

#[derive(Message)]
struct Opset {
    #[prost(int64, tag = "2")]
    version: i64,
}

#[derive(Message)]
struct Graph {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<Node>,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(message, repeated, tag = "5")]
    initializers: Vec<Initializer>,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfo>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfo>,
}

#[derive(Message)]
struct Node {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op: String,
    #[prost(message, repeated, tag = "5")]
    attrs: Vec<Attribute>,
}

#[derive(Message)]
pub(crate) struct Attribute {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(float, tag = "2")]
    float: f32,
    #[prost(int64, tag = "3")]
    int: i64,
    #[prost(bytes, tag = "4")]
    text: Vec<u8>,
    #[prost(int64, repeated, tag = "8")]
    ints: Vec<i64>,
    #[prost(int32, tag = "20")]
    kind: i32,
}

impl Attribute {
    pub(crate) fn int(name: &str, value: i64) -> Self {
        Self {
            name: name.into(),
            int: value,
            kind: 2,
            ..Self::default()
        }
    }

    pub(crate) fn ints(name: &str, values: &[i64]) -> Self {
        Self {
            name: name.into(),
            ints: values.to_owned(),
            kind: 7,
            ..Self::default()
        }
    }

    pub(crate) fn float(name: &str, value: f32) -> Self {
        Self {
            name: name.into(),
            float: value,
            kind: 1,
            ..Self::default()
        }
    }

    pub(crate) fn text(name: &str, value: &str) -> Self {
        Self {
            name: name.into(),
            text: value.as_bytes().to_owned(),
            kind: 3,
            ..Self::default()
        }
    }
}

#[derive(Message)]
struct Initializer {
    #[prost(int64, repeated, tag = "1")]
    dimensions: Vec<i64>,
    #[prost(int32, tag = "2")]
    kind: i32,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes, tag = "9")]
    raw_data: Vec<u8>,
    #[prost(message, repeated, tag = "13")]
    external_data: Vec<Pair>,
    #[prost(int32, tag = "14")]
    data_location: i32,
}

#[derive(Message)]
struct Pair {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

impl Pair {
    fn new(key: &str, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

#[derive(Message)]
struct ValueInfo {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    kind: Option<Type>,
}

#[derive(Message)]
struct Type {
    #[prost(message, optional, tag = "1")]
    tensor: Option<TensorType>,
}

#[derive(Message)]
struct TensorType {
    #[prost(int32, tag = "1")]
    element: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<Shape>,
}

#[derive(Message)]
struct Shape {
    #[prost(message, repeated, tag = "1")]
    dimensions: Vec<Dimension>,
}

#[derive(Message)]
pub(crate) struct Dimension {
    #[prost(int64, optional, tag = "1")]
    size: Option<i64>,
    #[prost(string, optional, tag = "2")]
    symbol: Option<String>,
}

impl Dimension {
    pub(crate) fn fixed(size: i64) -> Self {
        Self {
            size: Some(size),
            symbol: None,
        }
    }

    pub(crate) fn variable(symbol: &str) -> Self {
        Self {
            size: None,
            symbol: Some(symbol.into()),
        }
    }
}

impl ValueInfo {
    fn new(name: String, element: i32, dimensions: Vec<Dimension>) -> Self {
        let shape = Shape { dimensions };
        let tensor = TensorType {
            element,
            shape: Some(shape),
        };
        Self {
            name,
            kind: Some(Type {
                tensor: Some(tensor),
            }),
        }
    }
}

pub(crate) struct Builder {
    graph: Graph,
    weights: BufWriter<File>,
    written: u64,
}

impl Builder {
    pub(crate) fn new(directory: &Path, name: &str, shape: Vec<Dimension>) -> Result<Self> {
        Ok(Self {
            graph: Graph {
                name: name.into(),
                input: vec![ValueInfo::new("input".into(), FLOAT, shape)],
                ..Graph::default()
            },
            weights: BufWriter::new(File::create(directory.join(WEIGHTS))?),
            written: 0,
        })
    }

    pub(crate) fn input(&mut self, name: &str, shape: Vec<Dimension>) {
        self.graph
            .input
            .push(ValueInfo::new(name.to_owned(), FLOAT, shape));
    }

    pub(crate) fn integer_input(&mut self, name: &str, shape: Vec<Dimension>) {
        self.graph
            .input
            .push(ValueInfo::new(name.to_owned(), INT64, shape));
    }

    pub(crate) fn output(&mut self, value: String, shape: Vec<Dimension>) {
        self.graph.output.push(ValueInfo::new(value, FLOAT, shape));
    }

    pub(crate) fn op(&mut self, op: &str, inputs: &[&str], attrs: Vec<Attribute>) -> String {
        let ordinal = self.graph.nodes.len();
        let output = format!("value_{ordinal}");
        self.graph.nodes.push(Node {
            input: inputs.iter().map(|name| (*name).to_owned()).collect(),
            output: vec![output.clone()],
            name: format!("{op}_{ordinal}"),
            op: op.into(),
            attrs,
        });
        output
    }

    fn initializer(&mut self, mut value: Initializer) -> String {
        let name = format!("weight_{}", self.graph.initializers.len());
        value.name = name.clone();
        self.graph.initializers.push(value);
        name
    }

    pub(crate) fn tensor(&mut self, tensor: &Tensor) -> Result<String> {
        self.typed_tensor(tensor, Kind::Float, FLOAT, 4)
    }

    pub(crate) fn half_tensor(&mut self, tensor: &Tensor) -> Result<String> {
        self.typed_tensor(tensor, Kind::Half, FLOAT16, 2)
    }

    fn typed_tensor(
        &mut self,
        tensor: &Tensor,
        dtype: Kind,
        onnx_type: i32,
        bytes_per_value: usize,
    ) -> Result<String> {
        let tensor = tensor
            .f_to_device(Device::Cpu)?
            .f_to_kind(dtype)?
            .f_contiguous()?;
        let dimensions = tensor.size();
        let count = tensor.numel();
        let byte_count = count
            .checked_mul(bytes_per_value)
            .context("ONNX weight size overflow")?;
        let mut bytes = vec![0u8; byte_count];
        tensor.f_copy_data_u8(&mut bytes, count)?;
        #[cfg(target_endian = "big")]
        for bytes in bytes.chunks_exact_mut(bytes_per_value) {
            bytes.reverse();
        }
        self.weights.write_all(&bytes)?;
        let external_data = vec![
            Pair::new("location", WEIGHTS),
            Pair::new("offset", self.written.to_string()),
            Pair::new("length", byte_count.to_string()),
        ];
        self.written += byte_count as u64;
        Ok(self.initializer(Initializer {
            dimensions,
            kind: onnx_type,
            data_location: 1,
            external_data,
            ..Initializer::default()
        }))
    }

    pub(crate) fn integers(&mut self, shape: &[i64], values: &[i64]) -> String {
        self.initializer(Initializer {
            dimensions: shape.to_owned(),
            kind: INT64,
            raw_data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            ..Initializer::default()
        })
    }

    pub(crate) fn scale(&mut self, value: f32) -> String {
        self.floats(&[], &[value])
    }

    pub(crate) fn floats(&mut self, shape: &[i64], values: &[f32]) -> String {
        self.initializer(Initializer {
            dimensions: shape.to_owned(),
            kind: FLOAT,
            raw_data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            ..Initializer::default()
        })
    }

    pub(crate) fn finish(
        mut self,
        path: &Path,
        output: String,
        shape: Vec<Dimension>,
    ) -> Result<()> {
        self.weights.flush()?;
        self.output(output, shape);
        let model = Model {
            ir_version: 9,
            producer: "rvc".into(),
            graph: Some(self.graph),
            opsets: vec![Opset { version: 20 }],
        };
        std::fs::write(path, model.encode_to_vec())?;
        Ok(())
    }
}

pub(crate) struct Layers<'a> {
    pub(crate) graph: Builder,
    pub(crate) weights: &'a Weights,
}

impl Layers<'_> {
    pub(crate) fn transpose(&mut self, value: &str, axes: &[i64]) -> String {
        self.graph
            .op("Transpose", &[value], vec![Attribute::ints("perm", axes)])
    }

    pub(crate) fn conv(
        &mut self,
        value: &str,
        name: &str,
        stride: i64,
        pad: i64,
        groups: i64,
    ) -> Result<String> {
        let attrs = vec![
            Attribute::ints("strides", &[stride]),
            Attribute::ints("pads", &[pad, pad]),
            Attribute::int("group", groups),
        ];
        self.convolution("Conv", value, name, attrs)
    }

    pub(crate) fn convolution(
        &mut self,
        op: &str,
        value: &str,
        name: &str,
        attrs: Vec<Attribute>,
    ) -> Result<String> {
        let weight = self
            .graph
            .tensor(self.weights.get(&format!("{name}.weight"))?)?;
        let key = format!("{name}.bias");
        if self.weights.contains(&key) {
            let bias = self.graph.tensor(self.weights.get(&key)?)?;
            Ok(self.graph.op(op, &[value, &weight, &bias], attrs))
        } else {
            Ok(self.graph.op(op, &[value, &weight], attrs))
        }
    }

    pub(crate) fn linear(&mut self, value: &str, name: &str) -> Result<String> {
        let weight = self.graph.tensor(
            &self
                .weights
                .get(&format!("{name}.weight"))?
                .f_transpose(0, 1)?,
        )?;
        let value = self.graph.op("MatMul", &[value, &weight], vec![]);
        let key = format!("{name}.bias");
        if self.weights.contains(&key) {
            let bias = self.graph.tensor(self.weights.get(&key)?)?;
            Ok(self.graph.op("Add", &[&value, &bias], vec![]))
        } else {
            Ok(value)
        }
    }

    pub(crate) fn norm(&mut self, op: &str, value: &str, name: &str) -> Result<String> {
        let weight = self
            .graph
            .tensor(self.weights.get(&format!("{name}.weight"))?)?;
        let bias = self
            .graph
            .tensor(self.weights.get(&format!("{name}.bias"))?)?;
        Ok(self.graph.op(
            op,
            &[value, &weight, &bias],
            vec![Attribute::float("epsilon", 1e-5)],
        ))
    }
}
