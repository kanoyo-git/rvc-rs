use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::watch;
use uuid::Uuid;

use super::{Shared, WebResult, files, lock};

// A ticket is lightweight and holds no compute permit. A proxy may buffer the whole
// upload before contacting Studio, so the client needs a cancellable ID beforehand.
#[derive(Default)]
pub(super) struct Tickets(Mutex<HashMap<Uuid, Entry>>);

enum Entry {
    Offered(Instant),
    Reading(watch::Sender<bool>),
}

impl Tickets {
    fn issue(&self) -> Result<Option<Uuid>> {
        let mut entries = lock(&self.0)?;
        entries.retain(|_, entry| !matches!(entry, Entry::Offered(since) if since.elapsed() > Duration::from_secs(24 * 3600)));
        if entries.len() >= 256 {
            return Ok(None);
        }
        let id = Uuid::new_v4();
        entries.insert(id, Entry::Offered(Instant::now()));
        Ok(Some(id))
    }

    fn begin(self: &Arc<Self>, id: Uuid) -> Result<Option<Transfer>> {
        let mut entries = lock(&self.0)?;
        let Some(entry @ Entry::Offered(_)) = entries.get_mut(&id) else {
            return Ok(None);
        };
        let (send, cancelled) = watch::channel(false);
        *entry = Entry::Reading(send);
        Ok(Some(Transfer {
            id,
            tickets: Arc::clone(self),
            cancelled,
        }))
    }

    pub(super) fn revoke(&self, id: Uuid) -> Result<bool> {
        match lock(&self.0)?.remove(&id) {
            Some(Entry::Reading(send)) => {
                send.send_replace(true);
                Ok(true)
            }
            Some(Entry::Offered(_)) => Ok(true),
            None => Ok(false),
        }
    }
}

pub(super) struct Transfer {
    pub(super) id: Uuid,
    tickets: Arc<Tickets>,
    cancelled: watch::Receiver<bool>,
}
impl Transfer {
    pub(super) fn cancelled(&self) -> bool {
        *self.cancelled.borrow()
    }
}
impl Drop for Transfer {
    fn drop(&mut self) {
        self.tickets
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
    }
}

/// Atomically hand the ID to the accepted job (or publish a staged dataset). DELETE
/// either revokes the transfer or finds the admitted job with this very same ID.
pub(super) fn handoff<T>(
    transfer: Option<Transfer>,
    accept: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    let Some(transfer) = transfer else {
        return accept().map(Some);
    };
    let mut entries = lock(&transfer.tickets.0)?;
    if entries.remove(&transfer.id).is_none() {
        return Ok(None);
    }
    accept().map(Some)
}

pub(super) async fn ticket(State(state): State<Shared>) -> WebResult<Response> {
    Ok(match state.studio.tickets.issue()? {
        Some(id) => (StatusCode::CREATED, Json(json!({ "id": id }))).into_response(),
        None => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({ "error": "Too many pending uploads." })),
        )
            .into_response(),
    })
}

#[derive(Deserialize)]
pub(super) struct Query {
    pub(super) ticket: Option<Uuid>,
}

pub(super) struct Received {
    pub(super) settings: Value,
    pub(super) uploads: files::Uploads,
    pub(super) transfer: Option<Transfer>,
}

pub(super) async fn receive(
    state: &Shared,
    form: Multipart,
    id: Option<Uuid>,
    dataset: bool,
) -> Result<Option<Received>> {
    let reader = files::upload(state, form, dataset);
    let Some(id) = id else {
        let (settings, uploads) = reader.await?;
        return Ok(Some(Received {
            settings,
            uploads,
            transfer: None,
        }));
    };
    let Some(mut transfer) = state.studio.tickets.begin(id)? else {
        return Ok(None);
    };
    let (settings, uploads) = tokio::select! {
        biased;
        _ = transfer.cancelled.changed() => return Ok(None),
        result = reader => result?,
    };
    Ok(Some(Received {
        settings,
        uploads,
        transfer: Some(transfer),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_cancelled_or_disconnected_transfer_cannot_start_a_job() -> Result<()> {
        let tickets = Arc::new(Tickets::default());
        let early = tickets.issue()?.unwrap();
        assert!(tickets.revoke(early)?);
        assert!(tickets.begin(early)?.is_none());
        let id = tickets.issue()?.unwrap();
        let mut upload = tickets.begin(id)?.unwrap();
        assert!(tickets.begin(id)?.is_none());
        assert!(tickets.revoke(id)?);
        upload.cancelled.changed().await?;
        assert!(
            handoff(Some(upload), || -> Result<()> {
                panic!("revoked upload admitted")
            })?
            .is_none()
        );
        let id = tickets.issue()?.unwrap();
        drop(tickets.begin(id)?.unwrap());
        assert!(tickets.begin(id)?.is_none());
        let id = tickets.issue()?.unwrap();
        assert_eq!(handoff(tickets.begin(id)?, || Ok(id))?, Some(id));
        assert!(!tickets.revoke(id)?);
        Ok(())
    }
}
