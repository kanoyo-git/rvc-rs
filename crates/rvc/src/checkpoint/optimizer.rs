use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use tch::{Device, Kind, Tensor};

use super::{Checkpoint, Id, Value, storage_bytes};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AdamWGroup {
    pub lr: f64,
    pub betas: [f64; 2],
    pub eps: f64,
    pub weight_decay: f64,
    #[serde(default)]
    pub amsgrad: bool,
    #[serde(default)]
    pub maximize: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_lr: Option<f64>,
    pub params: Vec<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_names: Option<Vec<String>>,
    #[serde(flatten)]
    pub other: BTreeMap<String, Json>,
}

pub(crate) struct AdamWMoments {
    pub step: f64,
    pub step_kind: Kind,
    pub exp_avg: Tensor,
    pub exp_avg_sq: Tensor,
    pub max_exp_avg_sq: Option<Tensor>,
}

pub(crate) struct AdamWState {
    pub group: AdamWGroup,
    pub moments: BTreeMap<i64, AdamWMoments>,
}

impl Checkpoint {
    pub(crate) fn adamw_state(&mut self) -> Result<AdamWState> {
        let optimizer = self.dictionary(self.section(Some("optimizer"))?)?;
        let group_id = required(&optimizer, "param_groups")?;
        let state_id = required(&optimizer, "state")?;
        let Value::List(groups) = &self.values[group_id] else {
            bail!("AdamW param_groups is not a list");
        };
        let &[group] = groups.as_slice() else {
            bail!("RVC checkpoints have exactly one AdamW parameter group");
        };
        let group: AdamWGroup = serde_json::from_value(self.json(group, 0, &mut 16_384)?)
            .context("reading AdamW parameter group")?;
        let ids: BTreeSet<_> = group.params.iter().copied().collect();
        ensure!(
            !ids.is_empty() && ids.len() == group.params.len(),
            "invalid AdamW parameter identifiers"
        );
        let Value::Dict(state) = &self.values[state_id] else {
            bail!("AdamW state is not a dictionary");
        };
        let entries = state.clone();
        let mut budget = 2 * 1024 * 1024 * 1024;
        let mut moments = BTreeMap::new();
        for (key, value) in entries {
            let Value::Int(key) = self.values[key] else {
                bail!("AdamW state keys must be integers");
            };
            ensure!(
                ids.contains(&key),
                "AdamW state for unknown parameter {key}"
            );
            ensure!(
                !moments.contains_key(&key),
                "duplicate AdamW parameter state {key}"
            );
            let fields = self.dictionary(value)?;
            let step = required(&fields, "step")?;
            let exp_avg = required(&fields, "exp_avg")?;
            let exp_avg_sq = required(&fields, "exp_avg_sq")?;
            let max_exp_avg_sq = fields.get("max_exp_avg_sq").copied();
            let (step, step_kind) = self.adamw_step(step, &mut budget)?;
            let exp_avg = self.adamw_tensor(exp_avg, &mut budget)?;
            let exp_avg_sq = self.adamw_tensor(exp_avg_sq, &mut budget)?;
            let max_exp_avg_sq = max_exp_avg_sq
                .map(|id| self.adamw_tensor(id, &mut budget))
                .transpose()?;
            ensure!(
                group.amsgrad == max_exp_avg_sq.is_some(),
                "AMSGrad state and parameter group disagree for parameter {key}"
            );
            moments.insert(
                key,
                AdamWMoments {
                    step,
                    step_kind,
                    exp_avg,
                    exp_avg_sq,
                    max_exp_avg_sq,
                },
            );
        }
        Ok(AdamWState { group, moments })
    }

    fn adamw_step(&mut self, id: Id, budget: &mut u64) -> Result<(f64, Kind)> {
        let (step, kind) = match self.values[id] {
            Value::Int(step) => (step as f64, Kind::Float),
            Value::Float(step) => (step, Kind::Float),
            Value::Tensor(_) => {
                let tensor = self.adamw_tensor(id, budget)?;
                let kind = tensor.kind();
                ensure!(tensor.numel() == 1, "AdamW step must be a scalar");
                ensure!(
                    matches!(kind, Kind::Double | Kind::Float),
                    "unsupported AdamW step dtype {kind:?}"
                );
                (tensor.f_reshape([])?.f_double_value(&[])?, kind)
            }
            _ => bail!("invalid AdamW step value"),
        };
        ensure!(
            step.is_finite() && step >= 0.0 && step.fract() == 0.0,
            "invalid AdamW step {step}"
        );
        Ok((step, kind))
    }

    fn adamw_tensor(&mut self, id: Id, budget: &mut u64) -> Result<Tensor> {
        let Value::Tensor(reference) = &self.values[id] else {
            bail!("AdamW moment is not a tensor");
        };
        let reference = reference.clone();
        *budget = budget
            .checked_sub(storage_bytes(&reference.storage)?)
            .context("AdamW tensor storage exceeds import limits")?;
        self.tensor(&reference, Device::Cpu)
    }
}

fn required(fields: &BTreeMap<&str, Id>, name: &str) -> Result<Id> {
    fields
        .get(name)
        .copied()
        .with_context(|| format!("AdamW state is missing {name}"))
}
