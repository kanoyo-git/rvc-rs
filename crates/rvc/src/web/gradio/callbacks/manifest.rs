use std::collections::BTreeSet;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use rvc::dataset::{ManifestRowInput, manifest_from_rows, save_manifest};

use super::{Args, StateData, lock, training, update};

const PAGE_SIZE: usize = 10;
const MAX_ROWS: usize = 110;
const MAX_SESSIONS: usize = 256;

pub(in crate::web::gradio) struct Picker {
    rows: Vec<[Value; 4]>,
    active: usize,
    page: usize,
    touched: Instant,
}

impl Picker {
    fn new() -> Self {
        let mut rows = vec![blank(); MAX_ROWS];
        for (id, row) in rows[..2].iter_mut().enumerate() {
            *row = speaker_row(id);
        }
        Self {
            rows,
            active: 2,
            page: 0,
            touched: Instant::now(),
        }
    }

    fn save_page(&mut self, args: &Args, first: usize) {
        for (slot, row) in self.rows[self.page * PAGE_SIZE..]
            .iter_mut()
            .take(PAGE_SIZE)
            .enumerate()
        {
            for (column, value) in row.iter_mut().enumerate() {
                let index = first + slot * 4 + column;
                if let Some(input) = args.0.get(index) {
                    *value = input.clone();
                }
            }
        }
        self.touched = Instant::now();
    }

    fn apply(&mut self, index: usize) {
        match index {
            27 => {
                if self.active < MAX_ROWS {
                    let used: BTreeSet<_> = self.rows[..self.active]
                        .iter()
                        .filter_map(|row| row[2].as_u64().or_else(|| row[2].as_str()?.parse().ok()))
                        .collect();
                    let id = (0..MAX_ROWS)
                        .find(|&id| !used.contains(&(id as u64)))
                        .unwrap_or(0);
                    self.rows[self.active] = speaker_row(id);
                    self.active += 1;
                }
                self.page = (self.active - 1) / PAGE_SIZE;
            }
            28 => {
                if self.active > 2 {
                    self.active -= 1;
                    self.rows[self.active] = blank();
                }
            }
            29 => self.page = self.page.saturating_sub(1),
            30 => self.page += 1,
            _ => unreachable!(),
        }
        self.page = self.page.min((self.active - 1) / PAGE_SIZE);
    }

    fn updates(&self) -> Vec<Value> {
        // The first three outputs are Gradio State components.
        let mut result = vec![Value::Null; 3];
        let pages = self.active.div_ceil(PAGE_SIZE);
        result.push(json!(format!(
            "<p>Page {}/{pages}, {} rows</p>\n",
            self.page + 1,
            self.active
        )));
        for (slot, row) in self.rows[self.page * PAGE_SIZE..]
            .iter()
            .take(PAGE_SIZE)
            .enumerate()
        {
            let visible = self.page * PAGE_SIZE + slot < self.active;
            for (column, value) in row.iter().enumerate() {
                let value = if column >= 2 && value.as_str() == Some("") {
                    &Value::Null
                } else {
                    value
                };
                result.push(update(json!({"value":value,"visible":visible})));
            }
        }
        result
    }
}

fn blank() -> [Value; 4] {
    [json!(""), json!(""), json!(""), json!("")]
}

fn speaker_row(id: usize) -> [Value; 4] {
    [json!(""), json!(""), json!(id), json!(1)]
}

pub(super) fn run(
    state: &StateData,
    session: Option<&str>,
    index: usize,
    args: &Args,
) -> Result<Vec<Value>> {
    let session = session.unwrap_or_default();
    let mut sessions = lock(&state.legacy.manifests)?;
    if !sessions.contains_key(session)
        && sessions.len() >= MAX_SESSIONS
        && let Some(oldest) = sessions
            .iter()
            .min_by_key(|(_, state)| state.touched)
            .map(|(id, _)| id.clone())
    {
        sessions.remove(&oldest);
    }
    let picker = sessions
        .entry(session.to_owned())
        .or_insert_with(Picker::new);
    let submit = index == 31;
    picker.save_page(args, if submit { 4 } else { 3 });
    if !submit {
        picker.apply(index);
        return Ok(picker.updates());
    }
    let mut rows = Vec::with_capacity(picker.active);
    for [path, speaker_name, speaker_id, repeat] in &picker.rows[..picker.active] {
        rows.push(
            serde_json::from_value::<ManifestRowInput>(json!({
                "path":path, "speaker_name":speaker_name, "speaker_id":speaker_id, "repeat":repeat,
            }))
            .context("invalid speaker manifest row")?,
        );
    }
    drop(sessions);
    let experiment = training::experiment(state, args.text(0)?)?;
    let result = manifest_from_rows(&rows, Some(&state.paths.root))?;
    let path = save_manifest(&experiment, &result.manifest)?;
    let message = format!(
        "Multi-speaker training manifest saved to {} ({} audio files)",
        path.display(),
        result.manifest.entries.len(),
    );
    let message = message
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let suffix = if result.invalid_rows.is_empty() {
        String::new()
    } else {
        format!("<br>Ignored invalid rows {:?}", result.invalid_rows)
    };
    Ok(vec![
        Value::Null,
        json!(format!("<div role=\"status\">{message}{suffix}</div>")),
    ])
}
