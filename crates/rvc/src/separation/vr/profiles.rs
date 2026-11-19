use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Clone, Copy)]
pub(super) struct Metadata {
    pub primary: &'static str,
    pub secondary: &'static str,
    pub parameters: &'static str,
}

macro_rules! profile {
    ($primary:literal, $secondary:literal, $params:literal) => {
        Metadata {
            primary: $primary,
            secondary: $secondary,
            parameters: include_str!(concat!(
                "../../../resources/pymss/vr_modelparams/",
                $params,
                ".json"
            )),
        }
    };
}

pub(super) fn metadata(checkpoint: &Path) -> Option<Metadata> {
    Some(match checkpoint.file_name()?.to_str()? {
        "10_SP-UVR-2B-32000-1.pth" | "11_SP-UVR-2B-32000-2.pth" => {
            profile!("Instrumental", "Vocals", "2band_32000")
        }
        "12_SP-UVR-3B-44100.pth" => profile!("Instrumental", "Vocals", "3band_44100"),
        "13_SP-UVR-4B-44100-1.pth"
        | "14_SP-UVR-4B-44100-2.pth"
        | "1_HP-UVR.pth"
        | "8_HP2-UVR.pth"
        | "9_HP2-UVR.pth" => profile!("Instrumental", "Vocals", "4band_44100"),
        "15_SP-UVR-MID-44100-1.pth" | "16_SP-UVR-MID-44100-2.pth" => {
            profile!("Instrumental", "Vocals", "3band_44100_mid")
        }
        "17_HP-Wind_Inst-UVR.pth" => profile!("No Woodwinds", "Woodwinds", "4band_v3"),
        "2_HP-UVR.pth" => profile!("Instrumental", "Vocals", "4band_v2"),
        "3_HP-Vocal-UVR.pth" | "4_HP-Vocal-UVR.pth" => {
            profile!("Vocals", "Instrumental", "4band_44100")
        }
        "5_HP-Karaoke-UVR.pth" => profile!("Instrumental", "Vocals", "4band_v2_sn"),
        "6_HP-Karaoke-UVR.pth" | "7_HP2-UVR.pth" => {
            profile!("Instrumental", "Vocals", "3band_44100_msb2")
        }
        "Harmonic_Noise_Separation_yxlllc.pth" => {
            profile!("No Aspiration", "Aspiration", "1band_sr44100_hl1024")
        }
        "MGM_HIGHEND_v4.pth" => profile!("Instrumental", "Vocals", "1band_sr44100_hl1024"),
        "MGM_LOWEND_A_v4.pth" => profile!("Instrumental", "Vocals", "1band_sr32000_hl512"),
        "MGM_LOWEND_B_v4.pth" => profile!("Instrumental", "Vocals", "1band_sr33075_hl384"),
        "MGM_MAIN_v4.pth" => profile!("Instrumental", "Vocals", "1band_sr44100_hl512"),
        "UVR-BVE-4B_SN-44100-1.pth" => profile!("Vocals", "Instrumental", "4band_v3_sn"),
        "UVR-De-Echo-Aggressive.pth" | "UVR-De-Echo-Normal.pth" => {
            profile!("No Echo", "Echo", "4band_v3")
        }
        "UVR-DeEcho-DeReverb.pth" => profile!("No Reverb", "Reverb", "4band_v3"),
        "UVR-DeNoise-Lite.pth" => profile!("Noise", "No Noise", "1band_sr44100_hl1024"),
        "UVR-DeNoise.pth" => profile!("Noise", "No Noise", "4band_v3"),
        "UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband.pth" => {
            profile!("Dry", "Reverb", "4band_v4_ms_fullband")
        }
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Channels {
    #[default]
    Identity,
    Reverse,
    MidSide,
    MidSideB2,
    MidSideC,
    StereoN,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Resampler {
    Polyphase,
    SoxrHq,
    SincBest,
    SincMedium,
    SincFastest,
    KaiserFast,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(super) struct Band {
    #[serde(rename = "sr")]
    pub rate: u32,
    #[serde(rename = "hl")]
    pub hop: i64,
    #[serde(rename = "n_fft")]
    pub fft: i64,
    pub crop_start: i64,
    pub crop_stop: i64,
    #[serde(default)]
    pub hpf_start: i64,
    #[serde(default)]
    pub hpf_stop: i64,
    #[serde(default)]
    pub lpf_start: i64,
    #[serde(default)]
    pub lpf_stop: i64,
    #[serde(rename = "res_type")]
    pub resampler: Resampler,
    #[serde(default, rename = "convert_channels")]
    pub channels: Channels,
}

#[derive(Debug, Deserialize)]
pub(super) struct Correction {
    pub left: f32,
    pub right: f32,
}

#[derive(Debug, Deserialize)]
pub(super) struct Profile {
    #[serde(alias = "n_bins")]
    pub bins: i64,
    #[serde(rename = "sr")]
    pub rate: u32,
    pub pre_filter_start: i64,
    pub pre_filter_stop: i64,
    #[serde(rename = "band")]
    pub bands: BTreeMap<u32, Band>,
    pub aggr_correction: Option<Correction>,
    #[serde(default)]
    reverse: bool,
    #[serde(default)]
    mid_side: bool,
    #[serde(default)]
    mid_side_b2: bool,
}

impl Profile {
    pub(super) fn parse(document: &str) -> Result<Self> {
        let profile: Self =
            serde_json::from_str(document).context("reading VR frequency band profile")?;
        ensure!(
            profile.bins > 0 && profile.rate > 0,
            "invalid VR profile dimensions"
        );
        ensure!(
            !profile.bands.is_empty(),
            "VR profile has no frequency bands"
        );
        ensure!(
            profile
                .bands
                .keys()
                .copied()
                .eq(1..=profile.bands.len() as u32),
            "VR profile bands must be numbered consecutively"
        );
        let mut width = 0;
        for band in profile.bands.values() {
            ensure!(
                band.rate > 0
                    && band.hop > 0
                    && band.hop <= band.fft
                    && band.fft > 0
                    && band.fft % 2 == 0
                    && 0 <= band.crop_start
                    && band.crop_start < band.crop_stop
                    && band.crop_stop <= band.fft / 2 + 1,
                "invalid VR band geometry"
            );
            width += band.crop_stop - band.crop_start;
        }
        ensure!(
            width <= profile.bins,
            "VR model profile crops exceed the network bandwidth"
        );
        ensure!(
            profile.pre_filter_start <= profile.bins && profile.pre_filter_stop <= profile.bins,
            "invalid VR pre-filter frequencies"
        );
        Ok(profile)
    }

    pub(super) fn legacy_channels(&self) -> Channels {
        if self.reverse {
            Channels::Reverse
        } else if self.mid_side {
            Channels::MidSide
        } else if self.mid_side_b2 {
            Channels::MidSideB2
        } else {
            Channels::Identity
        }
    }
}
