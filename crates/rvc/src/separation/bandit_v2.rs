//! Bandit v2 configuration and state-dict layout.
//! See `licenses/pymss-core.txt`.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use tch::{Device, Tensor};

use super::bandit::{BandSpec, Bandit, Config, Version};

pub(super) fn load(
    document: &Value,
    weights: &HashMap<String, Tensor>,
    device: Device,
) -> Result<Bandit> {
    let mut config =
        Config::deserialize(&document["kwargs"]).context("invalid Bandit v2 config.kwargs")?;
    let name = config
        .band_type
        .take()
        .unwrap_or_else(|| "musical".to_owned());
    ensure!(
        name == "musical",
        "Bandit v2 requires musical frequency bands"
    );
    config.band_specs = Some(BandSpec::Named(name));
    config.n_bands = Some(config.n_bands.unwrap_or(64));
    Bandit::load(config, weights, device, Version::V2, document)
}
