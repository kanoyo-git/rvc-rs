use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use axum::http::{HeaderValue, Uri, uri::Authority};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;

const RELAY_REQUEST: &str = "https://api.gradio.app/v2/tunnel-request";

pub fn share_client() -> PathBuf {
    std::env::var_os("RVC_SHARE_CLIENT").map_or_else(|| "frpc-gradio".into(), PathBuf::from)
}

#[derive(Deserialize)]
struct Relay {
    host: String,
    port: u16,
}

impl Relay {
    fn address(&self) -> Result<String> {
        match self.host.parse::<IpAddr>() {
            Ok(ip) => Ok(SocketAddr::new(ip, self.port).to_string()),
            Err(_) => {
                ensure!(
                    self.host.split('.').all(|part| !part.is_empty()
                        && part.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')),
                    "invalid share relay address"
                );
                Ok(format!("{}:{}", self.host, self.port))
            }
        }
    }
}

async fn relay() -> Result<Relay> {
    let reply = Command::new("curl")
        .args([
            "-q",
            "--proto",
            "=https",
            "--max-time",
            "10",
            "--fail",
            "--silent",
            "--show-error",
            RELAY_REQUEST,
        ])
        .kill_on_drop(true)
        .output()
        .await
        .context("querying the Gradio share relay")?;
    ensure!(
        reply.status.success(),
        "querying the Gradio share relay: {}",
        String::from_utf8_lossy(&reply.stderr).trim(),
    );
    serde_json::from_slice::<Vec<Relay>>(&reply.stdout)?
        .into_iter()
        .find(|relay| relay.port > 0)
        .context("the Gradio share service returned no live relay")
}

#[derive(Clone)]
pub(super) struct Access {
    authority: Authority,
    authorization: HeaderValue,
}

impl Access {
    fn new(address: &str, password: &str) -> Result<Self> {
        let url = if address.starts_with("https://") {
            address.to_owned()
        } else if let Some(host) = address.strip_prefix("http://") {
            format!("https://{host}")
        } else {
            format!("https://{address}")
        };
        let uri: Uri = url
            .parse()
            .context("invalid share URL from the connector")?;
        let authority = uri.authority().cloned().context("missing share hostname")?;
        let host = authority.host().to_ascii_lowercase();
        ensure!(
            (host.ends_with(".gradio.live") || host.ends_with(".gradio.app"))
                && uri.path() == "/"
                && uri.query().is_none()
                && authority.port_u16().is_none()
                && !authority.as_str().contains('@'),
            "the share connector did not return a Gradio HTTPS origin"
        );
        let encoded = STANDARD.encode(format!("rvc:{password}"));
        let mut authorization = HeaderValue::from_str(&format!("Basic {encoded}"))?;
        authorization.set_sensitive(true);
        Ok(Self {
            authority,
            authorization,
        })
    }

    pub(super) fn authority(&self) -> &Authority {
        &self.authority
    }

    pub(super) fn authorized(&self, presented: Option<&HeaderValue>) -> bool {
        presented.is_some_and(|candidate| {
            let expected = self.authorization.as_bytes();
            let candidate = candidate.as_bytes();
            expected.len() == candidate.len()
                && expected
                    .iter()
                    .zip(candidate)
                    .fold(0, |difference, (a, b)| difference | (a ^ b))
                    == 0
        })
    }
}

pub(super) struct Tunnel {
    connector: Connector,
    access: Access,
    generated_password: Option<String>,
}

impl Tunnel {
    pub(super) async fn connect(client: &Path, listener: SocketAddr) -> Result<Self> {
        let configured = std::env::var("RVC_SHARE_PASSWORD").ok();
        let password = configured
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        ensure!(!password.is_empty(), "RVC_SHARE_PASSWORD is empty");
        let relay = relay().await?.address()?;
        let (connector, origin) = Connector::open(client, listener, &relay).await?;
        let access = Access::new(&origin, &password)?;
        Ok(Self {
            connector,
            access,
            generated_password: configured.is_none().then_some(password),
        })
    }

    pub(super) fn access(&self) -> Access {
        self.access.clone()
    }

    pub(super) fn announce(&self, panel: &str) {
        println!("https://{}/{panel}", self.access.authority);
        if let Some(password) = &self.generated_password {
            println!("RVC share login rvc, password {password}");
        } else {
            println!("RVC share login rvc, password from RVC_SHARE_PASSWORD");
        }
    }

    pub(super) async fn wait_for_exit(&mut self) {
        match self.connector.child.wait().await {
            Ok(exit) => eprintln!("RVC share connector exited with {exit}"),
            Err(error) => eprintln!("RVC share connector wait failed: {error}"),
        }
    }

    pub(super) async fn stop(&mut self) {
        let _ = self.connector.child.kill().await;
    }
}

struct Connector {
    child: Child,
    output: JoinHandle<()>,
}

impl Connector {
    async fn open(client: &Path, listener: SocketAddr, relay: &str) -> Result<(Self, String)> {
        let mut child = Command::new(client)
            .args([
                "http",
                "-n",
                "random",
                "-l",
                &listener.port().to_string(),
                "-i",
                &listener.ip().to_string(),
                "--uc",
                "--sd",
                "random",
                "--ue",
                "--server_addr",
                relay,
                "--disable_log_color",
            ])
            .stdout(Stdio::piped())
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!("starting Gradio-compatible connector {}", client.display())
            })?;
        let stdout = child
            .stdout
            .take()
            .context("share connector has no output pipe")?;
        let mut lines = BufReader::new(stdout).lines();
        let mut diagnostic = None;
        let origin = tokio::time::timeout(Duration::from_secs(30), async {
            while let Some(line) = lines
                .next_line()
                .await
                .context("reading share connector output")?
            {
                if let Some((_, address)) = line.split_once("start proxy success:")
                    && !address.trim().is_empty()
                {
                    return Ok(address.trim().to_owned());
                }
                if !line.trim().is_empty() {
                    diagnostic = Some(line);
                }
            }
            let status = child
                .wait()
                .await
                .context("waiting for the share connector")?;
            anyhow::bail!("share connector exited with {status}");
        })
        .await
        .context("timed out waiting for the share connector")
        .and_then(std::convert::identity);
        let origin = match origin {
            Ok(origin) => origin,
            Err(error) => {
                let _ = child.kill().await;
                let reason = "the share connector did not report a public URL";
                return Err(match diagnostic {
                    Some(line) => error.context(format!("{reason}: {line}")),
                    None => error.context(reason),
                });
            }
        };
        let output =
            tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
        Ok((Self { child, output }, origin))
    }
}

impl Drop for Connector {
    fn drop(&mut self) {
        self.output.abort();
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    // A concurrent fork can hold a writable fixture descriptor until exec.
    static CONNECTOR_FIXTURES: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn client_script(dir: &Path, body: &str) -> PathBuf {
        let client = dir.join("connector");
        fs::write(&client, format!("#!/bin/sh\n{body}")).unwrap();
        fs::set_permissions(&client, fs::Permissions::from_mode(0o700)).unwrap();
        client
    }

    #[tokio::test]
    async fn cancelling_startup_stops_the_connector() {
        let _fixture = CONNECTOR_FIXTURES.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let client = client_script(
            dir.path(),
            "printf '%s\\n' \"$$\" > \"$0.pid\"\nexec sleep 30\n",
        );
        let pid_file = dir.path().join("connector.pid");
        let mut startup = Box::pin(Connector::open(
            &client,
            "127.0.0.1:7865".parse().unwrap(),
            "127.0.0.1:7000",
        ));
        let pid: u32 = tokio::select! {
            _ = &mut startup => panic!("stalled connector exited before cancellation"),
            pid = tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    if let Ok(value) = tokio::fs::read_to_string(&pid_file).await
                        && let Ok(pid) = value.trim().parse()
                    {
                        break pid;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }) => pid.expect("connector did not start"),
        };
        let process = PathBuf::from(format!("/proc/{pid}"));
        assert!(process.is_dir());
        drop(startup);
        tokio::time::timeout(Duration::from_secs(3), async {
            while process.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("connector survived cancellation");
    }

    #[tokio::test]
    async fn startup_failure_reports_the_connector_diagnostic() {
        let _fixture = CONNECTOR_FIXTURES.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let client = client_script(
            dir.path(),
            "echo 'login to server failed: connection refused'\nexit 1\n",
        );
        let result =
            Connector::open(&client, "127.0.0.1:7865".parse().unwrap(), "127.0.0.1:7000").await;
        let error = result.err().expect("connector unexpectedly started");
        let message = format!("{error:#}");
        assert!(
            message.contains("login to server failed: connection refused"),
            "{message}"
        );
    }
}
