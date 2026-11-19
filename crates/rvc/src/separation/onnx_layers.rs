use anyhow::Result;
use tch::Tensor;

use crate::model::onnx::graph::{Attribute, Builder};

pub(super) fn linear(
    graph: &mut Builder,
    value: &str,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<String> {
    let weight = graph.tensor(&weight.f_transpose(0, 1)?)?;
    let value = graph.op("MatMul", &[value, &weight], vec![]);
    if let Some(bias) = bias {
        let bias = graph.tensor(bias)?;
        Ok(graph.op("Add", &[&value, &bias], vec![]))
    } else {
        Ok(value)
    }
}

pub(super) fn convolution(
    graph: &mut Builder,
    op: &str,
    input: &str,
    weights: &Tensor,
    bias: Option<&Tensor>,
    geometry: Vec<Attribute>,
) -> Result<String> {
    let weights = graph.tensor(weights)?;
    if let Some(bias) = bias {
        let bias = graph.tensor(bias)?;
        Ok(graph.op(op, &[input, &weights, &bias], geometry))
    } else {
        Ok(graph.op(op, &[input, &weights], geometry))
    }
}

pub(super) fn transpose(graph: &mut Builder, value: &str, axes: &[i64]) -> String {
    graph.op("Transpose", &[value], vec![Attribute::ints("perm", axes)])
}

pub(super) fn unsqueeze(graph: &mut Builder, value: &str, axes: &[i64]) -> String {
    let axes = graph.integers(&[axes.len() as i64], axes);
    graph.op("Unsqueeze", &[value, &axes], vec![])
}

pub(super) fn reshape(graph: &mut Builder, value: &str, shape: &[i64]) -> String {
    let shape = graph.integers(&[shape.len() as i64], shape);
    graph.op("Reshape", &[value, &shape], vec![])
}

pub(super) fn slice(
    graph: &mut Builder,
    value: &str,
    axis: i64,
    start: i64,
    end: i64,
    step: i64,
) -> String {
    let start = graph.integers(&[1], &[start]);
    let end = graph.integers(&[1], &[end]);
    let axis = graph.integers(&[1], &[axis]);
    let step = graph.integers(&[1], &[step]);
    graph.op("Slice", &[value, &start, &end, &axis, &step], vec![])
}

pub(super) fn select(graph: &mut Builder, value: &str, axis: i64, index: i64) -> String {
    let index = graph.integers(&[], &[index]);
    graph.op(
        "Gather",
        &[value, &index],
        vec![Attribute::int("axis", axis)],
    )
}

pub(super) fn dimensions(graph: &mut Builder, value: &str, start: i64, end: i64) -> String {
    graph.op(
        "Shape",
        &[value],
        vec![Attribute::int("start", start), Attribute::int("end", end)],
    )
}

pub(super) fn concat(graph: &mut Builder, values: &[&str], axis: i64) -> String {
    graph.op("Concat", values, vec![Attribute::int("axis", axis)])
}

pub(super) fn concat_strings(graph: &mut Builder, values: &[String], axis: i64) -> String {
    let values = values.iter().map(String::as_str).collect::<Vec<_>>();
    concat(graph, &values, axis)
}

pub(super) fn reduce(
    graph: &mut Builder,
    op: &str,
    value: &str,
    axes: &[i64],
    keep: bool,
) -> String {
    let axes = graph.integers(&[axes.len() as i64], axes);
    graph.op(
        op,
        &[value, &axes],
        vec![Attribute::int("keepdims", i64::from(keep))],
    )
}

pub(super) fn swish(graph: &mut Builder, value: &str) -> String {
    let gate = graph.op("Sigmoid", &[value], vec![]);
    graph.op("Mul", &[value, &gate], vec![])
}

pub(super) fn glu(graph: &mut Builder, value: &str, axis: i64, width: i64) -> String {
    let left = slice(graph, value, axis, 0, width, 1);
    let right = slice(graph, value, axis, width, width * 2, 1);
    let right = graph.op("Sigmoid", &[&right], vec![]);
    graph.op("Mul", &[&left, &right], vec![])
}

pub(super) enum Recurrent {
    Lstm,
    Gru,
    Tanh,
}

pub(super) fn recurrent(
    graph: &mut Builder,
    value: &str,
    params: &[Tensor],
    hidden: i64,
    bidirectional: bool,
    kind: Recurrent,
) -> Result<String> {
    let (operator, gate_order): (&str, &[i64]) = match kind {
        Recurrent::Lstm => ("LSTM", &[0, 3, 1, 2]),
        Recurrent::Gru => ("GRU", &[1, 0, 2]),
        Recurrent::Tanh => ("RNN", &[0]),
    };
    let reorder = |weight: &Tensor| {
        let gates = gate_order
            .iter()
            .map(|gate| weight.f_narrow(0, gate * hidden, hidden))
            .collect::<Result<Vec<_>, _>>()?;
        Tensor::f_cat(&gates, 0)
    };
    let directions = 1 + i64::from(bidirectional);
    let parameter = |i| -> Result<Tensor> {
        let values = params
            .as_chunks::<4>()
            .0
            .iter()
            .map(|weights| reorder(&weights[i]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tensor::f_stack(&values, 0)?)
    };
    let input_weight = graph.tensor(&parameter(0)?)?;
    let hidden_weight = graph.tensor(&parameter(1)?)?;
    let bias = graph.tensor(&Tensor::f_cat(&[parameter(2)?, parameter(3)?], 1)?)?;
    let sequence = transpose(graph, value, &[1, 0, 2]);
    let mut attrs = vec![
        Attribute::text(
            "direction",
            if bidirectional {
                "bidirectional"
            } else {
                "forward"
            },
        ),
        Attribute::int("hidden_size", hidden),
    ];
    if matches!(kind, Recurrent::Gru) {
        attrs.push(Attribute::int("linear_before_reset", 1));
    }
    let sequence = graph.op(
        operator,
        &[&sequence, &input_weight, &hidden_weight, &bias],
        attrs,
    );
    let sequence = transpose(graph, &sequence, &[2, 0, 1, 3]);
    Ok(reshape(graph, &sequence, &[0, 0, directions * hidden]))
}
