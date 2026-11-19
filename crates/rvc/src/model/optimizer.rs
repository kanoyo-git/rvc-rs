use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};
use serde_json::{Value as Json, json};
use tch::{Device, Kind, Tensor};

use crate::checkpoint::optimizer::{AdamWGroup, AdamWMoments, AdamWState};
use crate::checkpoint::write::{Key, Value};

pub(crate) struct AdamW {
    group: AdamWGroup,
    parameters: Vec<Parameter>,
}

struct Parameter {
    name: String,
    value: Tensor,
    moments: Option<AdamWMoments>,
}

pub(super) struct GradientStep {
    pub overflow: bool,
    pub norm: f64,
}

impl AdamW {
    /// The caller supplies trainable leaves in PyTorch parameter registration order.
    pub(crate) fn new(
        parameters: Vec<(String, Tensor)>,
        lr: f64,
        betas: [f64; 2],
        eps: f64,
    ) -> Result<Self> {
        let count = i64::try_from(parameters.len())?;
        ensure!(count > 0, "AdamW needs trainable parameters");
        let group = AdamWGroup {
            lr,
            betas,
            eps,
            weight_decay: 0.01,
            amsgrad: false,
            maximize: false,
            initial_lr: None,
            params: (0..count).collect(),
            param_names: None,
            other: BTreeMap::from([
                ("capturable".to_owned(), Json::Bool(false)),
                ("differentiable".to_owned(), Json::Bool(false)),
                ("decoupled_weight_decay".to_owned(), Json::Bool(true)),
                ("foreach".to_owned(), Json::Null),
                ("fused".to_owned(), Json::Null),
            ]),
        };
        validate_hyperparameters(&group)?;
        let mut names = BTreeSet::new();
        let parameters = parameters
            .into_iter()
            .map(|(name, value)| {
                ensure!(
                    !name.is_empty() && names.insert(name.clone()),
                    "duplicate or empty trainable parameter name {name:?}"
                );
                ensure!(
                    is_float(value.kind()) && value.requires_grad() && value.f_is_leaf()?,
                    "AdamW parameter {name} must be a floating-point autograd leaf"
                );
                Ok(Parameter {
                    name,
                    value,
                    moments: None,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { group, parameters })
    }

    pub(crate) fn learning_rate(&self) -> f64 {
        self.group.lr
    }

    pub(super) fn gradients(&self, loss: &Tensor, scale: Option<f32>) -> Result<Vec<Tensor>> {
        let parameters: Vec<_> = self
            .parameters
            .iter()
            .map(|parameter| &parameter.value)
            .collect();
        backward(loss, &parameters, scale)
    }

    pub(crate) fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        self.step(self.gradients(loss, None)?, None, false)?;
        Ok(())
    }

    pub(super) fn step(
        &mut self,
        mut gradients: Vec<Tensor>,
        scale: Option<f32>,
        log_norm: bool,
    ) -> Result<GradientStep> {
        ensure!(
            gradients.len() == self.parameters.len(),
            "AdamW gradient count differs from parameters"
        );
        tch::no_grad(|| {
            let overflow = scale
                .map(|factor| unscale(&mut gradients, factor))
                .transpose()?
                .unwrap_or(false);
            let norm = if log_norm {
                gradient_norm(&gradients)?
            } else {
                0.0
            };
            if !overflow {
                for (parameter, gradient) in self.parameters.iter_mut().zip(gradients) {
                    if gradient.defined() {
                        parameter.update(gradient, &self.group).with_context(|| {
                            format!("updating AdamW parameter {}", parameter.name)
                        })?;
                    }
                }
            }
            Ok(GradientStep { overflow, norm })
        })
    }

    pub(crate) fn restore(&mut self, state: AdamWState) -> Result<()> {
        let AdamWState { group, mut moments } = state;
        validate_hyperparameters(&group)?;
        ensure!(
            group.params.len() == self.parameters.len(),
            "optimizer checkpoint has {} parameter identifiers for {} trainable leaves",
            group.params.len(),
            self.parameters.len()
        );
        if let Some(names) = &group.param_names {
            ensure!(
                names
                    .iter()
                    .map(String::as_str)
                    .eq(self.parameters.iter().map(|p| p.name.as_str())),
                "optimizer checkpoint parameter names differ from the current model"
            );
        }
        let loaded = self
            .parameters
            .iter()
            .zip(&group.params)
            .map(|(parameter, id)| {
                moments
                    .remove(id)
                    .map(|mut moment| {
                        moment.exp_avg = restore_tensor(moment.exp_avg, parameter, "exp_avg")?;
                        moment.exp_avg_sq =
                            restore_tensor(moment.exp_avg_sq, parameter, "exp_avg_sq")?;
                        moment.max_exp_avg_sq = moment
                            .max_exp_avg_sq
                            .map(|tensor| restore_tensor(tensor, parameter, "max_exp_avg_sq"))
                            .transpose()?;
                        Ok(moment)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            moments.is_empty(),
            "optimizer checkpoint contains unmatched parameter state"
        );
        self.group = group;
        for (parameter, moments) in self.parameters.iter_mut().zip(loaded) {
            parameter.moments = moments;
        }
        Ok(())
    }

    pub(crate) fn state_dict(&self) -> Result<Value> {
        let mut moments = Vec::new();
        for (id, parameter) in self.group.params.iter().zip(&self.parameters) {
            let Some(state) = &parameter.moments else {
                continue;
            };
            let step = Tensor::f_from_slice(&[state.step])?
                .f_to_kind(state.step_kind)?
                .f_reshape([])?;
            let mut fields = vec![
                ("step".into(), Value::Tensor(step)),
                (
                    "exp_avg".into(),
                    Value::Tensor(state.exp_avg.shallow_clone()),
                ),
                (
                    "exp_avg_sq".into(),
                    Value::Tensor(state.exp_avg_sq.shallow_clone()),
                ),
            ];
            if let Some(maximum) = &state.max_exp_avg_sq {
                fields.push((
                    "max_exp_avg_sq".into(),
                    Value::Tensor(maximum.shallow_clone()),
                ));
            }
            moments.push((Key::Integer(*id), Value::Dict(fields)));
        }
        let Json::Object(mut group) = serde_json::to_value(&self.group)? else {
            anyhow::bail!("optimizer parameter group must serialize as an object");
        };
        group.remove("betas");
        let mut fields: Vec<_> = group
            .into_iter()
            .map(|(key, value)| (Key::String(key), Value::Meta(value)))
            .collect();
        fields.push((
            "betas".into(),
            Value::Tuple(
                self.group
                    .betas
                    .iter()
                    .map(|value| Value::Meta(json!(value)))
                    .collect(),
            ),
        ));
        Ok(Value::Dict(vec![
            ("state".into(), Value::Dict(moments)),
            (
                "param_groups".into(),
                Value::List(vec![Value::Dict(fields)]),
            ),
        ]))
    }
}

pub(super) fn backward(
    loss: &Tensor,
    parameters: &[&Tensor],
    scale: Option<f32>,
) -> Result<Vec<Tensor>> {
    ensure!(loss.numel() == 1, "AdamW loss must be scalar");
    if let Some(factor) = scale {
        let scaled = loss.f_mul_scalar(f64::from(factor))?;
        Ok(Tensor::f_run_backward(
            &[&scaled],
            parameters,
            false,
            false,
        )?)
    } else {
        Ok(Tensor::f_run_backward(&[loss], parameters, false, false)?)
    }
}

fn unscale(gradients: &mut [Tensor], factor: f32) -> Result<bool> {
    let Some(first) = gradients.iter().find(|gradient| gradient.defined()) else {
        return Ok(false);
    };
    let device = first.device();
    let reciprocal = f64::from(factor).recip() as f32;
    if matches!(device, Device::Cuda(_)) {
        let options = (Kind::Float, device);
        let mut found = Tensor::f_zeros([], options)?;
        let inverse = Tensor::f_full([], f64::from(reciprocal), options)?;
        for gradient in gradients.iter_mut().filter(|tensor| tensor.defined()) {
            *gradient = gradient.f_contiguous()?;
            gradient.f_internal_amp_non_finite_check_and_unscale(&mut found, &inverse)?;
        }
        return Ok(found.f_double_value(&[])? != 0.0);
    }
    let mut overflow = false;
    for gradient in gradients.iter_mut().filter(|tensor| tensor.defined()) {
        overflow |= gradient.f_isfinite()?.f_all()?.f_int64_value(&[])? == 0;
        *gradient = gradient.f_mul_scalar(f64::from(reciprocal))?;
    }
    Ok(overflow)
}

fn gradient_norm(gradients: &[Tensor]) -> Result<f64> {
    let norms = gradients
        .iter()
        .filter(|gradient| gradient.defined())
        .map(Tensor::f_norm)
        .collect::<tch::Result<Vec<_>>>()?;
    if norms.is_empty() {
        return Ok(0.0);
    }
    let norms = Tensor::f_stack(&norms, 0)?
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Double)?;
    Ok(Vec::<f64>::try_from(norms)?
        .into_iter()
        .map(|n| n * n)
        .sum::<f64>()
        .sqrt())
}

impl Parameter {
    fn update(&mut self, mut gradient: Tensor, group: &AdamWGroup) -> Result<()> {
        ensure!(!gradient.is_sparse(), "AdamW requires dense gradients");
        if group.maximize {
            gradient = gradient.f_neg()?;
        }
        if self.moments.is_none() {
            self.moments = Some(AdamWMoments {
                step: 0.0,
                step_kind: Kind::Float,
                exp_avg: self.value.f_zeros_like()?,
                exp_avg_sq: self.value.f_zeros_like()?,
                max_exp_avg_sq: group
                    .amsgrad
                    .then(|| self.value.f_zeros_like())
                    .transpose()?,
            });
        }
        let moments = self
            .moments
            .as_mut()
            .context("initializing AdamW moments")?;
        moments.step = if moments.step_kind == Kind::Double {
            moments.step + 1.0
        } else {
            f64::from(moments.step as f32 + 1.0)
        };
        if group.weight_decay != 0.0 {
            let _ = self
                .value
                .f_mul_scalar_(1.0 - group.lr * group.weight_decay)?;
        }
        let [beta1, beta2] = group.betas;
        let _ = moments.exp_avg.f_lerp_(&gradient, 1.0 - beta1)?;
        let _ = moments.exp_avg_sq.f_mul_scalar_(beta2)?;
        let scaled_gradient = gradient.f_mul_scalar(1.0 - beta2)?;
        let _ = moments.exp_avg_sq.f_addcmul_(&scaled_gradient, &gradient)?;
        let second_moment = if let Some(maximum) = &mut moments.max_exp_avg_sq {
            *maximum = maximum.f_maximum(&moments.exp_avg_sq)?;
            maximum
        } else {
            &moments.exp_avg_sq
        };
        let step_size = group.lr / (1.0 - beta1.powf(moments.step));
        let variance_correction = (1.0 - beta2.powf(moments.step)).sqrt();
        let denominator = second_moment
            .f_sqrt()?
            .f_div_scalar(variance_correction)?
            .f_add_scalar(group.eps)?;
        let mean = moments.exp_avg.f_mul_scalar(-step_size)?;
        let _ = self.value.f_addcdiv_(&mean, &denominator)?;
        Ok(())
    }
}

fn restore_tensor(tensor: Tensor, parameter: &Parameter, field: &str) -> Result<Tensor> {
    ensure!(
        is_float(tensor.kind()) && tensor.size() == parameter.value.size(),
        "AdamW {field} does not match parameter {}",
        parameter.name
    );
    Ok(tensor
        .f_to_kind(parameter.value.kind())?
        .f_to_device(parameter.value.device())?)
}

fn is_float(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::Float | Kind::Double | Kind::Half | Kind::BFloat16
    )
}

fn validate_hyperparameters(group: &AdamWGroup) -> Result<()> {
    let valid_rate = |rate: f64| rate.is_finite() && rate >= 0.0;
    ensure!(
        valid_rate(group.lr)
            && group.initial_lr.is_none_or(valid_rate)
            && valid_rate(group.eps)
            && valid_rate(group.weight_decay)
            && group.betas.iter().all(|beta| (0.0..1.0).contains(beta)),
        "invalid AdamW hyperparameters"
    );
    for field in ["capturable", "differentiable"] {
        ensure!(
            group
                .other
                .get(field)
                .is_none_or(|value| value.as_bool() == Some(false)),
            "unsupported AdamW parameter group option {field}"
        );
    }
    Ok(())
}

pub(crate) struct ExponentialLr {
    gamma: f64,
}

impl ExponentialLr {
    pub(crate) fn new(optimizer: &mut AdamW, gamma: f64, first_epoch: usize) -> Result<Self> {
        ensure!(
            first_epoch > 0 && gamma.is_finite() && gamma >= 0.0,
            "invalid exponential schedule"
        );
        let schedule = Self { gamma };
        if first_epoch == 1 {
            optimizer.group.initial_lr.get_or_insert(optimizer.group.lr);
        } else {
            // RVC initializes ExponentialLR with last_epoch = first_epoch - 2.
            ensure!(
                optimizer.group.initial_lr.is_some(),
                "optimizer checkpoint lacks initial_lr"
            );
            schedule.step(optimizer)?;
        }
        Ok(schedule)
    }

    pub(crate) fn step(&self, optimizer: &mut AdamW) -> Result<()> {
        let learning_rate = optimizer.group.lr * self.gamma;
        ensure!(
            learning_rate.is_finite(),
            "exponential learning rate overflow"
        );
        optimizer.group.lr = learning_rate;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{Checkpoint, write};

    #[test]
    fn skips_optimizer_updates_when_scaled_gradients_overflow() -> Result<()> {
        let weight = Tensor::f_from_slice(&[0.4f32, -0.85])?.f_set_requires_grad(true)?;
        let mut optimizer = AdamW::new(
            vec![("weight".into(), weight.shallow_clone())],
            0.02,
            [0.8, 0.99],
            1e-9,
        )?;
        let loss = weight.f_sum(Kind::Float)?;
        let scale = Some(65_536.0);
        let first = optimizer.step(optimizer.gradients(&loss, scale)?, scale, true)?;
        assert!(!first.overflow);
        assert!((first.norm - 2f64.sqrt()).abs() < 1e-6);
        let saved = Vec::<f32>::try_from(&weight)?;

        let loss = weight.f_mul_scalar(f64::INFINITY)?.f_sum(Kind::Float)?;
        let overflow = optimizer.step(optimizer.gradients(&loss, scale)?, scale, false)?;
        assert!(overflow.overflow);
        assert_eq!(Vec::<f32>::try_from(&weight)?, saved);
        assert_eq!(optimizer.parameters[0].moments.as_ref().unwrap().step, 1.0);
        Ok(())
    }

    #[test]
    fn skips_unused_parameters_and_round_trips_optimizer_moments() -> Result<()> {
        let weight = Tensor::f_from_slice(&[0.4f32, -0.85])?.f_set_requires_grad(true)?;
        let bias = Tensor::f_from_slice(&[0.2f32])?.f_set_requires_grad(true)?;
        let mut optimizer = AdamW::new(
            vec![
                ("weight".into(), weight.shallow_clone()),
                ("bias".into(), bias.shallow_clone()),
            ],
            0.02,
            [0.8, 0.99],
            1e-9,
        )?;
        let scheduler = ExponentialLr::new(&mut optimizer, 0.9, 1)?;
        optimizer.backward_step(&weight.f_square()?.f_sum(Kind::Float)?)?;
        let updated = Vec::<f32>::try_from(&weight)?;
        assert!((updated[0] - 0.37992).abs() < 1e-6);
        assert!((updated[1] + 0.82983).abs() < 1e-6);
        scheduler.step(&mut optimizer)?;
        optimizer.backward_step(&bias.f_square()?.f_sum(Kind::Float)?)?;
        assert_eq!(Vec::<f32>::try_from(&weight)?, updated);
        assert!((bias.f_double_value(&[0])? - 0.181964).abs() < 1e-6);

        let directory = tempfile::tempdir()?;
        let snapshot = directory.path().join("optim.pth");
        let state = Value::Dict(vec![("optimizer".into(), optimizer.state_dict()?)]);
        write::write(&snapshot, &state, false)?;
        let mut archive = Checkpoint::open(&snapshot)?;
        let state = archive.adamw_state()?;
        assert_eq!(state.group.params, [0, 1]);
        assert_eq!(state.group.initial_lr, Some(0.02));
        assert_eq!(state.moments[&0].step, 1.0);
        assert_eq!(state.moments[&0].exp_avg.size(), [2]);
        assert_eq!(state.moments[&1].step, 1.0);
        optimizer.restore(state)?;
        ExponentialLr::new(&mut optimizer, 0.9, 2)?;
        assert!((optimizer.learning_rate() - 0.0162).abs() < 1e-15);
        Ok(())
    }
}
