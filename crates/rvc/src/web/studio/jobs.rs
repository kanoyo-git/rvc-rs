use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use axum::Json;
use axum::extract::{Path as RoutePath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::NamedTempFile;
use uuid::Uuid;

use super::super::{Job, JobState, Shared, WebResult, lock};

const RECENT_JOBS: usize = 8;
const PAGE_RECORDS: usize = 128;
const PAGE_BYTES: u64 = 128 * 1024;

#[derive(Clone, Serialize)]
pub(in crate::web) struct Metadata {
    output: &'static str,
    stop: &'static str,
    name: Option<String>,
    detail: Option<String>,
}

impl Metadata {
    pub(super) fn from_settings(operation: &str, settings: &Value) -> Result<Self> {
        let (output, stop) = match operation {
            "single" => ("sLog", "stopSingle"),
            "batch" => ("bLog", "stopBatch"),
            "pitch_profile" => ("iProfileLog", "iStopProfile"),
            "cover" => ("cLog", "stopCover"),
            "separate" => ("pLog", "stopSep"),
            "pretrained"
                if settings["generator"].is_null() && settings["discriminator"].is_null() =>
            {
                ("trainLog", "stopPretrained")
            }
            "import" | "pretrained" => ("libLog", "stopLib"),
            "preprocess" => ("preLog", "stopPre"),
            "extract" => ("extLog", "stopExt"),
            "install_embedder" => ("extLog", "stopEmbedder"),
            "build_feature_pool" => ("trainLog", "stopFeaturePool"),
            "train" => ("trainLog", "stopTrain"),
            "index" => ("trainLog", "stopIndex"),
            "train_all" => ("trainLog", "stopAll"),
            "resume" => ("resumeLog", "stopResume"),
            "layer_aug" => ("laLog", "stopLayers"),
            "fx_profile" => ("fxLog", "stopFx"),
            _ => anyhow::bail!("unknown job operation {operation}"),
        };
        let name = match operation {
            "resume" => settings["name"].as_str(),
            "train" | "train_all" => settings["experiment"]
                .as_str()
                .and_then(|path| Path::new(path).file_name()?.to_str())
                .or_else(|| settings["name"].as_str()),
            _ => None,
        };
        let detail = match operation {
            "install_embedder" => settings["embedder"].as_str(),
            "layer_aug" | "fx_profile" => settings["output"].as_str(),
            _ => None,
        };
        Ok(Self {
            output,
            stop,
            name: name.map(str::to_owned),
            detail: detail.map(str::to_owned),
        })
    }
}

pub(in crate::web) struct Journal {
    metadata: Metadata,
    started_at: u64,
    file: Arc<NamedTempFile>,
    committed: u64,
    pub(in crate::web) error: Option<String>,
}

impl Journal {
    pub(in crate::web) fn new(metadata: Metadata, files: &super::Files) -> Result<Self> {
        let file =
            NamedTempFile::new_in(files.workspace.path()).context("creating Studio job log")?;
        Ok(Self {
            metadata,
            started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_millis()
                .try_into()?,
            file: Arc::new(file),
            committed: 0,
            error: None,
        })
    }

    pub(in crate::web) fn append(&mut self, message: &str) -> Result<()> {
        let mut record = serde_json::to_vec(message)?;
        record.push(b'\n');
        self.file
            .as_file()
            .write_all(&record)
            .context("writing Studio job log")?;
        self.committed += record.len() as u64;
        Ok(())
    }

    fn summary(&self, job: &Job) -> Summary {
        Summary {
            id: job.id,
            operation: job.operation.clone(),
            status: job.status,
            started_at: self.started_at,
            studio: self.metadata.clone(),
        }
    }
}

#[derive(Serialize)]
pub(super) struct Summary {
    id: Uuid,
    operation: String,
    status: JobState,
    started_at: u64,
    studio: Metadata,
}

pub(in crate::web) fn retain_recent(recent: &mut VecDeque<Job>, current: &Job) {
    let output = |job: &Job| job.studio.as_ref().map(|journal| journal.metadata.output);
    retain_recent_panels(recent, RECENT_JOBS, output(current), output);
}

pub(super) fn retain_recent_panels<T, K: Eq + std::hash::Hash>(
    entries: &mut VecDeque<T>,
    limit: usize,
    current: Option<K>,
    panel: impl Fn(&T) -> Option<K>,
) {
    if entries.len() <= limit {
        return;
    }
    let mut seen = HashSet::new();
    seen.extend(current);
    let mut keep = VecDeque::with_capacity(entries.len());
    for (age, entry) in entries.drain(..).rev().enumerate() {
        let latest_for_panel = panel(&entry).is_some_and(|key| seen.insert(key));
        if age < limit || latest_for_panel {
            keep.push_front(entry);
        }
    }
    *entries = keep;
}

pub(super) async fn list(State(state): State<Shared>) -> WebResult<Json<Vec<Summary>>> {
    let current = lock(&state.job)?;
    let recent = lock(&state.recent_jobs)?;
    let mut outputs = HashSet::new();
    let mut summaries = Vec::new();
    for job in std::iter::once(&*current).chain(recent.iter().rev()) {
        if let Some(journal) = &job.studio
            && outputs.insert(journal.metadata.output)
        {
            summaries.push(journal.summary(job));
        }
    }
    summaries.reverse();
    Ok(Json(summaries))
}

#[derive(Deserialize)]
pub(super) struct Position {
    cursor: Option<u64>,
}

pub(super) async fn status(
    State(state): State<Shared>,
    RoutePath(id): RoutePath<Uuid>,
    Query(query): Query<Position>,
) -> WebResult<Response> {
    let (summary, message, result, file, end, cursor) = {
        let current = lock(&state.job)?;
        let recent = lock(&state.recent_jobs)?;
        let Some(job) = std::iter::once(&*current)
            .chain(recent.iter())
            .find(|job| job.id == id)
        else {
            return Ok(StatusCode::NOT_FOUND.into_response());
        };
        let Some(cursor) = query.cursor else {
            return Ok(Json(job).into_response());
        };
        let Some(journal) = &job.studio else {
            return Ok(StatusCode::NOT_FOUND.into_response());
        };
        (
            journal.summary(job),
            job.message.clone(),
            job.result.clone(),
            Arc::clone(&journal.file),
            journal.committed,
            cursor,
        )
    };
    tokio::task::spawn_blocking(move || {
        let (log, next_cursor) = page(&file, end, cursor)?;
        let more = next_cursor < end;
        #[derive(Serialize)]
        struct Page<'a> {
            #[serde(flatten)]
            summary: Summary,
            message: String,
            log: Vec<String>,
            next_cursor: u64,
            more: bool,
            result: Option<&'a Value>,
        }
        Ok(Json(Page {
            summary,
            message,
            log,
            next_cursor,
            more,
            result: if more { None } else { result.as_deref() },
        })
        .into_response())
    })
    .await
    .map_err(anyhow::Error::from)?
}

fn page(file: &NamedTempFile, end: u64, cursor: u64) -> Result<(Vec<String>, u64)> {
    ensure!(cursor <= end, "job log cursor is past the end");
    if cursor == end {
        return Ok((Vec::new(), cursor));
    }
    let mut file = file.reopen().context("opening Studio job log")?;
    if cursor > 0 {
        file.seek(SeekFrom::Start(cursor - 1))?;
        let mut last = [0u8];
        file.read_exact(&mut last)?;
        ensure!(
            last == [b'\n'],
            "job log cursor is not at a record boundary"
        );
    }
    let mut reader = BufReader::new(file.take(end - cursor));
    let mut next = cursor;
    let mut lines = Vec::new();
    let mut record = Vec::new();
    while next < end && lines.len() < PAGE_RECORDS && next - cursor < PAGE_BYTES {
        record.clear();
        let bytes = reader.read_until(b'\n', &mut record)? as u64;
        ensure!(
            record.last() == Some(&b'\n'),
            "incomplete Studio job log record"
        );
        // A single long message is always delivered intact.
        if !lines.is_empty() && next - cursor + bytes > PAGE_BYTES {
            break;
        }
        lines.push(serde_json::from_slice(&record).context("reading Studio job log entry")?);
        next += bytes;
    }
    Ok((lines, next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn job(files: &super::super::Files, operation: &str) -> Result<Job> {
        Ok(Job {
            id: Uuid::new_v4(),
            operation: operation.to_owned(),
            status: JobState::Complete,
            studio: Some(Journal::new(
                Metadata::from_settings(operation, &Value::Null)?,
                files,
            )?),
            ..Default::default()
        })
    }

    #[test]
    fn cursors_preserve_multiline_utf8_and_bound_pages_during_append() -> Result<()> {
        let files = super::super::Files::new()?;
        let mut job = job(&files, "train")?;
        let journal = job.studio.as_mut().unwrap();
        let mut expected = vec![
            "Начало\nновая строка".to_owned(),
            "д".repeat(PAGE_BYTES as usize),
        ];
        expected.extend((0..300).map(|i| format!("эпоха {i}")));
        for line in &expected {
            journal.append(line)?;
        }
        let end = journal.committed;
        journal.append("after snapshot")?;
        let mut result = Vec::new();
        let mut cursor = 0;
        while cursor < end {
            let (lines, next) = page(&journal.file, end, cursor)?;
            assert!(next > cursor && lines.len() <= PAGE_RECORDS);
            assert!(next - cursor <= PAGE_BYTES || lines.len() == 1);
            result.extend(lines);
            cursor = next;
        }
        assert_eq!(expected, result);
        assert_eq!(
            page(&journal.file, journal.committed, cursor)?.0,
            ["after snapshot"]
        );
        assert!(page(&journal.file, end, 2).is_err());
        assert!(page(&journal.file, end, end + 1).is_err());
        Ok(())
    }

    #[test]
    fn keeps_eight_recent_jobs_and_latest_for_each_output() -> Result<()> {
        let files = super::super::Files::new()?;
        let conversion = job(&files, "single")?;
        let conversion_id = conversion.id;
        let previous = job(&files, "train")?;
        let previous_id = previous.id;
        let previous_log = previous.studio.as_ref().unwrap().file.path().to_owned();
        let mut recent = VecDeque::from([conversion, previous]);
        let eight: Vec<_> = (0..8)
            .map(|_| job(&files, "extract"))
            .collect::<Result<_>>()?;
        let ids: Vec<_> = eight.iter().map(|j| j.id).collect();
        recent.extend(eight);
        retain_recent(&mut recent, &job(&files, "index")?);
        assert_eq!(recent.len(), 9);
        assert_eq!(recent[0].id, conversion_id);
        assert!(recent.iter().all(|job| job.id != previous_id));
        assert!(!previous_log.exists(), "eviction must release the old log");
        assert!(recent.iter().skip(1).map(|j| j.id).eq(ids));
        assert_eq!(
            Metadata::from_settings("pretrained", &json!({}))?.output,
            "trainLog"
        );
        assert_eq!(
            Metadata::from_settings("pretrained", &json!({"generator":"url"}))?.output,
            "libLog"
        );
        Ok(())
    }
}
