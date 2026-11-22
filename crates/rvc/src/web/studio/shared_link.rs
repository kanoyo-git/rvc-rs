//! Resolve public file-sharing links, then save one verified transfer into the caller's staging area.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs::File;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use percent_encoding::percent_decode_str;
use reqwest::cookie::{CookieStore, Jar};
use reqwest::header::{CONTENT_DISPOSITION, CONTENT_TYPE};
use reqwest::{Client, Response, Url};
use scraper::{ElementRef, Html, Selector};
use serde::Deserialize;
use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;
use tokio_util::compat::TokioAsyncWriteCompatExt;

const MAX_FILE: u64 = 4 * 1024 * 1024 * 1024;
const MAX_PAGE: u64 = 2 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(3600);

static FORMS: LazyLock<Selector> = LazyLock::new(|| Selector::parse("form").unwrap());
static INPUTS: LazyLock<Selector> = LazyLock::new(|| Selector::parse("input[name]").unwrap());
static LINKS: LazyLock<Selector> =
    LazyLock::new(|| Selector::parse("a[href], a[data-scrambled-url]").unwrap());

pub(super) fn download(
    raw: &str,
    output: &Path,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Option<String>> {
    let url = http_url(raw.trim())?;
    ensure!(
        report("Downloading shared file").is_continue(),
        "download cancelled"
    );
    let parent = output
        .parent()
        .context("download destination has no directory")?;
    let staged = NamedTempFile::new_in(parent)?;
    let cookies = Arc::new(Jar::default());
    let client = Client::builder()
        .cookie_provider(cookies.clone())
        .user_agent("NVC-Studio/1.0")
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(60))
        .timeout(TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(8))
        .build()?;
    let total = Cell::new(0u64);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let name = runtime.block_on(async {
        let transfer = async {
            for attempt in 0..3 {
                let file = staged.reopen()?;
                file.set_len(0)?;
                total.set(0);
                let result = download_once(&client, &cookies, url.clone(), file, &total).await;
                if attempt == 2 || !result.as_ref().is_err_and(transient) {
                    return result;
                }
                tokio::time::sleep(Duration::from_secs(1 << attempt)).await;
            }
            unreachable!()
        };
        let transfer = tokio::time::timeout(TIMEOUT, transfer);
        tokio::pin!(transfer);
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        loop {
            tokio::select! {
                result = &mut transfer => return result.context("shared file download timed out")?,
                _ = tick.tick() => {
                    let done = staged.as_file().metadata()?.len();
                    let size = total.get();
                    let mib = done as f64 / 1_048_576.0;
                    let line = if size > 0 {
                        format!("Downloading shared file · {:.0}% · {mib:.1} / {:.1} MiB", 100.0 * done as f64 / size as f64, size as f64 / 1_048_576.0)
                    } else {
                        format!("Downloading shared file · {mib:.1} MiB")
                    };
                    ensure!(report(&line).is_continue(), "download cancelled");
                }
            }
        }
    })?;
    ensure!(
        report("Checking downloaded file").is_continue(),
        "download cancelled"
    );
    staged.persist(output).context("saving downloaded file")?;
    Ok(name)
}

fn transient(error: &anyhow::Error) -> bool {
    error.downcast_ref::<reqwest::Error>().is_some_and(|e| {
        e.is_connect()
            || e.is_timeout()
            || e.is_body()
            || e.status().is_some_and(|status| {
                status.is_server_error() || matches!(status.as_u16(), 408 | 429)
            })
    })
}

fn http_url(raw: &str) -> Result<Url> {
    let url = Url::parse(raw).context("invalid download link")?;
    ensure!(
        matches!(url.scheme(), "http" | "https") && url.host().is_some(),
        "use an HTTP(S) download link"
    );
    Ok(url)
}

async fn get(client: &Client, url: Url) -> Result<Response> {
    let res = client
        .get(url)
        .send()
        .await
        .map_err(reqwest::Error::without_url)?;
    Ok(res
        .error_for_status()
        .map_err(reqwest::Error::without_url)?)
}

fn disallow_large(response: &Response, limit: u64) -> Result<()> {
    ensure!(
        response.content_length().unwrap_or(0) <= limit,
        "download exceeds the size limit ({limit} bytes)"
    );
    Ok(())
}

async fn small_body(mut response: Response) -> Result<Vec<u8>> {
    disallow_large(&response, MAX_PAGE)?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(reqwest::Error::without_url)?
    {
        ensure!(
            (bytes.len() + chunk.len()) as u64 <= MAX_PAGE,
            "file-sharing page is too large"
        );
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

fn is_html(response: &Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            matches!(
                v.split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase()
                    .as_str(),
                "text/html" | "application/xhtml+xml"
            )
        })
}

fn safe_name(value: &str) -> Option<String> {
    let name = value.rsplit(['/', '\\']).next()?.trim();
    if name.is_empty()
        || matches!(name, "." | "..")
        || name.ends_with('.')
        || name.len() > 255
        || name.chars().any(char::is_control)
        || name.contains([':', '*', '?', '"', '<', '>', '|'])
    {
        return None;
    }
    Some(name.to_owned())
}

fn path_name(url: &Url) -> Option<String> {
    let name = url.path_segments()?.next_back()?;
    safe_name(&percent_decode_str(name).decode_utf8().ok()?)
}

fn disposition_name(header: &str) -> Option<String> {
    static EXTENDED: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r#"(?i)(?:^|;)\s*filename\*\s*=\s*"?utf-8'[^']*'([^";]+)"?"#).unwrap()
    });
    static SIMPLE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r#"(?i)(?:^|;)\s*filename\s*=\s*(?:"([^"]+)"|([^;]+))"#).unwrap()
    });
    EXTENDED
        .captures(header)
        .and_then(|c| safe_name(&percent_decode_str(c.get(1)?.as_str()).decode_utf8().ok()?))
        .or_else(|| {
            SIMPLE
                .captures(header)
                .and_then(|c| safe_name(c.get(1).or_else(|| c.get(2))?.as_str()))
        })
}

fn yandex(host: &str) -> bool {
    matches!(
        host,
        "yadi.sk"
            | "disk.yandex.ru"
            | "disk.yandex.com"
            | "disk.yandex.by"
            | "disk.yandex.kz"
            | "disk.yandex.ua"
            | "disk.360.yandex.ru"
    )
}

fn media_host(host: &str) -> bool {
    host == "mediafire.com" || host.ends_with(".mediafire.com")
}

fn google_host(host: &str) -> bool {
    matches!(
        host,
        "drive.google.com" | "docs.google.com" | "drive.usercontent.google.com"
    )
}

async fn download_once(
    client: &Client,
    cookies: &Jar,
    mut url: Url,
    file: File,
    total: &Cell<u64>,
) -> Result<Option<String>> {
    let host = url
        .host_str()
        .unwrap_or("")
        .trim_start_matches("www.")
        .to_owned();
    if matches!(host.as_str(), "mega.nz" | "mega.co.nz" | "mega.io") {
        return mega_file(client, &url, file, total).await;
    }
    url.set_fragment(None);
    let original_name = path_name(&url);
    let mut suggested = None;
    let response = if google_host(&host) {
        google(client, cookies, &url).await?
    } else if yandex(&host) {
        #[derive(Deserialize)]
        struct Link {
            href: String,
        }
        let mut api = Url::parse("https://cloud-api.yandex.net/v1/disk/public/resources/download")?;
        api.query_pairs_mut()
            .append_pair("public_key", url.as_str());
        let link: Link = serde_json::from_slice(&small_body(get(client, api).await?).await?)
            .context("Yandex Disk did not return a public file download")?;
        let direct = http_url(&link.href)?;
        suggested = direct
            .query_pairs()
            .find(|(key, _)| key == "filename")
            .and_then(|(_, v)| safe_name(&v));
        get(client, direct).await?
    } else if host == "mediafire.com" || host == "mfi.re" {
        let page = get(client, url).await?;
        if is_html(&page) {
            let base = page.url().clone();
            let body = small_body(page).await?;
            let direct = media_link(&base, &String::from_utf8_lossy(&body))
                .context("MediaFire did not offer a public file download")?;
            suggested = path_name(&direct);
            get(client, direct).await?
        } else {
            page
        }
    } else {
        if matches!(host.as_str(), "huggingface.co" | "hf.co") {
            let path = url.path().replacen("/blob/", "/resolve/", 1);
            url.set_path(&path);
            suggested = path_name(&url);
        }
        get(client, url).await?
    };
    ensure!(
        !is_html(&response),
        "the link returned a web page; choose a publicly downloadable file"
    );
    disallow_large(&response, MAX_FILE)?;
    let name = response
        .headers()
        .get(CONTENT_DISPOSITION)
        .and_then(|h| std::str::from_utf8(h.as_bytes()).ok())
        .and_then(disposition_name)
        .or(suggested)
        .or_else(|| path_name(response.url()).filter(|name| name.contains('.')))
        .or_else(|| original_name.filter(|name| name.contains('.')));
    let expected = response.content_length();
    total.set(expected.unwrap_or(0));
    let mut target = tokio::fs::File::from_std(file);
    let mut received = 0u64;
    let mut response = response;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(reqwest::Error::without_url)?
    {
        received += chunk.len() as u64;
        ensure!(received <= MAX_FILE, "download exceeds the 4 GiB limit");
        target.write_all(&chunk).await?;
    }
    target.flush().await?;
    ensure!(
        received > 0 && expected.is_none_or(|size| size == received),
        "download is empty or incomplete"
    );
    Ok(name)
}

fn drive_id(url: &Url) -> Option<String> {
    let id = url
        .query_pairs()
        .find(|(k, _)| k == "id")
        .map(|(_, v)| v.into_owned())
        .or_else(|| {
            let mut segments = url.path_segments()?;
            segments.find(|x| *x == "d")?;
            segments.next().map(str::to_owned)
        })?;
    (!id.is_empty()
        && id
            .bytes()
            .all(|x| x.is_ascii_alphanumeric() || matches!(x, b'_' | b'-')))
    .then_some(id)
}

async fn google(client: &Client, cookies: &Jar, shared: &Url) -> Result<Response> {
    let id = drive_id(shared).context("choose a Google Drive link to a file")?;
    let key = shared
        .query_pairs()
        .find(|(k, _)| k == "resourcekey")
        .map(|(_, v)| v.into_owned());
    let mut initial = Url::parse("https://drive.google.com/uc")?;
    initial
        .query_pairs_mut()
        .append_pair("id", &id)
        .append_pair("export", "download");
    if let Some(key) = &key {
        initial.query_pairs_mut().append_pair("resourcekey", key);
    }
    let response = get(client, initial.clone()).await?;
    if !is_html(&response) {
        return Ok(response);
    }
    let page = response.url().clone();
    let cookie = [&page, &initial]
        .into_iter()
        .filter_map(|url| cookies.cookies(url))
        .find_map(|h| {
            h.to_str().ok()?.split(';').find_map(|p| {
                let (name, v) = p.trim().split_once('=')?;
                name.starts_with("download_warning").then(|| v.to_owned())
            })
        });
    let html = small_body(response).await?;
    let url = drive_confirmation(
        &page,
        &String::from_utf8_lossy(&html),
        &id,
        key.as_deref(),
        cookie.as_deref(),
    )
    .context(
        "Google Drive did not offer a download; make the file accessible to anyone with its link",
    )?;
    get(client, url).await
}

fn drive_confirmation(
    page: &Url,
    html: &str,
    id: &str,
    key: Option<&str>,
    cookie: Option<&str>,
) -> Option<Url> {
    let document = Html::parse_document(html);
    let trusted = |url: &Url| url.scheme() == "https" && url.host_str().is_some_and(google_host);
    let finalize = |mut url: Url, mut fields: BTreeMap<String, String>| {
        if !trusted(&url) {
            return None;
        }
        for (k, v) in url.query_pairs() {
            fields
                .entry(k.into_owned())
                .or_insert_with(|| v.into_owned());
        }
        if !fields.contains_key("confirm") {
            fields.insert("confirm".into(), cookie?.to_owned());
        }
        fields.insert("id".into(), id.to_owned());
        fields.insert("export".into(), "download".into());
        if let Some(key) = key {
            fields.insert("resourcekey".into(), key.to_owned());
        }
        url.query_pairs_mut().clear().extend_pairs(fields);
        Some(url)
    };
    for form in document.select(&FORMS) {
        let fields: BTreeMap<_, _> = form
            .select(&INPUTS)
            .filter_map(|node| {
                Some((
                    node.value().attr("name")?.to_owned(),
                    node.value().attr("value").unwrap_or("").to_owned(),
                ))
            })
            .collect();
        let Some(action) = form.value().attr("action").and_then(|u| page.join(u).ok()) else {
            continue;
        };
        if let Some(url) = finalize(action, fields) {
            return Some(url);
        }
    }
    for link in document.select(&LINKS) {
        if let Some(url) = link.value().attr("href").and_then(|v| page.join(v).ok())
            && url.query_pairs().any(|(k, _)| k == "confirm")
            && let Some(confirmed) = finalize(url, BTreeMap::new())
        {
            return Some(confirmed);
        }
    }
    finalize(
        Url::parse("https://drive.usercontent.google.com/download").ok()?,
        BTreeMap::new(),
    )
}

fn media_link(page: &Url, html: &str) -> Option<Url> {
    let document = Html::parse_document(html);
    let resolve = |element: ElementRef<'_>| {
        let tag = element.value();
        let encoded = tag
            .attr("data-scrambled-url")
            .and_then(|raw| {
                STANDARD
                    .decode(raw)
                    .or_else(|_| URL_SAFE_NO_PAD.decode(raw))
                    .ok()
            })
            .and_then(|bytes| String::from_utf8(bytes).ok());
        [encoded.as_deref(), tag.attr("href")]
            .into_iter()
            .flatten()
            .find_map(|raw| {
                let url = page.join(raw).ok()?;
                let host = url.host_str()?;
                (matches!(url.scheme(), "http" | "https")
                    && media_host(host)
                    && (host.starts_with("download")
                        || (tag.id() == Some("downloadButton") && url != *page && raw != "#")))
                    .then_some(url)
            })
    };
    document
        .select(&LINKS)
        .filter(|el| el.value().id() == Some("downloadButton"))
        .find_map(resolve)
        .or_else(|| document.select(&LINKS).find_map(resolve))
}

fn mega_url(shared: &Url) -> Result<String> {
    let fragment = shared
        .fragment()
        .context("MEGA file link is missing its decryption key")?;
    let (handle, key) = if let Some(handle) = shared.path().strip_prefix("/file/") {
        (handle, fragment)
    } else {
        fragment
            .strip_prefix('!')
            .and_then(|s| s.split_once('!'))
            .context("choose a MEGA public file link")?
    };
    ensure!(
        !handle.is_empty()
            && handle.len() <= 32
            && handle
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_')),
        "invalid MEGA file handle"
    );
    let bytes = URL_SAFE_NO_PAD
        .decode(key)
        .context("invalid MEGA file key")?;
    ensure!(
        bytes.len() == 32 && bytes[..16] != bytes[16..],
        "invalid MEGA file key"
    );
    Ok(format!("https://mega.nz/file/{handle}#{key}"))
}

async fn mega_file(
    client: &Client,
    url: &Url,
    file: File,
    total: &Cell<u64>,
) -> Result<Option<String>> {
    let link = mega_url(url)?;
    let api = mega::Client::builder()
        .https(true)
        .max_retries(2)
        .timeout(Some(Duration::from_secs(60)))
        .build(client.clone())?;
    download_mega_node(api, link, file, total).await
}

async fn download_mega_node(
    api: mega::Client,
    link: String,
    file: File,
    total: &Cell<u64>,
) -> Result<Option<String>> {
    // The SDK asserts the decrypted attributes' magic bytes. A mistyped key must end the job
    // cleanly. Dropping this set also cancels the metadata request when a user cancels the job.
    let mut metadata = tokio::task::JoinSet::new();
    metadata.spawn(async move {
        let nodes = api.fetch_public_nodes(&link).await?;
        Ok::<_, mega::Error>((api, nodes))
    });
    let (api, nodes) = metadata
        .join_next()
        .await
        .context("missing MEGA metadata request")?
        .map_err(|_| anyhow::anyhow!("invalid MEGA file attributes or decryption key"))?
        .context("could not open the public MEGA file")?;
    let node = nodes.roots().next().context("MEGA did not return a file")?;
    ensure!(
        node.size() > 0 && node.size() <= MAX_FILE,
        "invalid MEGA file size (4 GiB maximum)"
    );
    total.set(node.size());
    let mut target = tokio::fs::File::from_std(file);
    // The SDK decrypts the stream and checks the full-coverage condensed MAC before returning.
    api.download_node(node, (&mut target).compat_write())
        .await
        .context("MEGA download or integrity check failed")?;
    target.flush().await?;
    ensure!(
        target.metadata().await?.len() == node.size(),
        "incomplete MEGA file"
    );
    Ok(safe_name(node.name()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    #[test]
    fn provider_links_and_names() -> Result<()> {
        let drive = http_url("https://drive.google.com/file/d/a_B9-z/view?resourcekey=K")?;
        assert_eq!(drive_id(&drive).as_deref(), Some("a_B9-z"));
        assert!(drive_id(&http_url("https://drive.google.com/open?id=%2Fetc")?).is_none());
        let form = r#"<form action="https://untrusted.test/collect"><input name="confirm" value="bad"></form><form action="https://drive.usercontent.google.com/download"><input name="confirm" value="yes"><input name="uuid" value="a&amp;b"></form>"#;
        let confirm = drive_confirmation(&drive, form, "a_B9-z", Some("K"), None).unwrap();
        assert_eq!(confirm.host_str(), Some("drive.usercontent.google.com"));
        let params = confirm.query_pairs().collect::<BTreeMap<_, _>>();
        assert_eq!(
            (
                params["id"].as_ref(),
                params["resourcekey"].as_ref(),
                params["uuid"].as_ref()
            ),
            ("a_B9-z", "K", "a&b")
        );
        assert!(drive_confirmation(&drive, "", "id", None, None).is_none());
        assert!(
            drive_confirmation(&drive, "", "id", None, Some("token"))
                .unwrap()
                .query_pairs()
                .any(|(k, v)| k == "confirm" && v == "token")
        );
        let media = http_url("https://www.mediafire.com/file/ABC/file")?;
        let direct = "https://download1.mediafire.com/abc/voice.zip";
        let media_page = format!(
            r##"<a href="https://ads.test/banner">Free</a><a id="downloadButton" href="#" data-scrambled-url="{}">Download</a>"##,
            STANDARD.encode(direct)
        );
        assert_eq!(
            media_link(&media, &media_page).as_ref().map(Url::as_str),
            Some(direct)
        );
        assert!(
            media_link(
                &media,
                r#"<a id="downloadButton" href="https://notmediafire.com/file.zip">"#
            )
            .is_none()
        );
        assert_eq!(
            disposition_name("attachment; filename=old.zip; filename*=UTF-8''..%2F%D0%B0.zip")
                .as_deref(),
            Some("а.zip")
        );
        assert_eq!(
            disposition_name(r#"attachment; filename="sample 2.zip""#).as_deref(),
            Some("sample 2.zip")
        );
        assert!(disposition_name("attachment; filename=..").is_none());
        Ok(())
    }

    // Returns a loopback origin. Each response is served on its own HTTP connection.
    fn serve(
        requests: usize,
        handler: impl Fn(&str, &str) -> (String, Vec<u8>) + Send + 'static,
    ) -> String {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let server_origin = origin.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(requests) {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(&stream);
                let mut lines = Vec::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    lines.push(line);
                }
                let size = lines
                    .iter()
                    .filter_map(|l| l.split_once(':'))
                    .find(|(k, _)| k.eq_ignore_ascii_case("Content-Length"))
                    .and_then(|(_, v)| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                reader.read_exact(&mut vec![0; size]).unwrap();
                let path = lines[0].split_whitespace().nth(1).unwrap();
                let (headers, body) = handler(path, &server_origin);
                write!(stream, "HTTP/1.1 {headers}\r\nConnection: close\r\n\r\n").unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        origin
    }

    #[test]
    fn direct_redirect_download_and_atomic_errors() -> Result<()> {
        let origin = serve(4, |path, _| match path {
            "/go" => ("302 Found\r\nLocation: /my%20voice.zip".into(), vec![]),
            "/my%20voice.zip" => ("200 OK\r\nContent-Length: 4".into(), b"ABCD".to_vec()),
            "/large" => (
                format!("200 OK\r\nContent-Length: {}", MAX_FILE + 1),
                vec![],
            ),
            _ => (
                "200 OK\r\nContent-Type: text/html".into(),
                b"<html>sign in</html>".to_vec(),
            ),
        });
        let dir = tempfile::tempdir()?;
        let output = dir.path().join("staging");
        let mut keep = |_: &str| ControlFlow::Continue(());
        assert_eq!(
            download(&format!("{origin}/go"), &output, &mut keep)?.as_deref(),
            Some("my voice.zip")
        );
        assert_eq!(std::fs::read(&output)?, b"ABCD");
        for (path, message) in [("large", "size limit"), ("sign-in", "web page")] {
            let err = download(&format!("{origin}/{path}"), &output, &mut keep).unwrap_err();
            assert!(err.to_string().contains(message), "{err:#}");
            assert_eq!(std::fs::read(&output)?, b"ABCD");
        }
        assert!(download(&origin, &output, &mut |_| ControlFlow::Break(())).is_err());
        assert_eq!(std::fs::read_dir(dir.path())?.count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn public_mega_stream_checks_file_mac_and_decodes_name() -> Result<()> {
        // Independently encrypted fixture: AES-128-CTR body, AES-128-CBC attributes and condensed MAC.
        const KEY: &str = "MTMxNzEzMT9iIYNPm6ZDHDEyMzQ1Njc4aiiJRJerTRM";
        let link = mega_url(&http_url(&format!("https://mega.nz/#!test1234!{KEY}"))?)?;
        assert_eq!(link, format!("https://mega.nz/file/test1234#{KEY}"));
        assert!(mega_url(&http_url("https://mega.nz/file/test1234#short")?).is_err());
        for corrupt in [false, true] {
            let origin = serve(3, move |path, origin| {
                let body = if path.starts_with("/cs?") {
                    serde_json::to_vec(&serde_json::json!([{"g":format!("{origin}/payload"), "s":17, "at":"ziHCcIgVF8IvKsuk217bvmIiYYanMFdn7vkaMSph7Mo"}])).unwrap()
                } else {
                    assert_eq!(path, "/payload/0-17");
                    let mut data = STANDARD.decode("9uuecWxV9LYcnMN2UI+rgB0=").unwrap();
                    if corrupt {
                        data[3] ^= 1;
                    }
                    data
                };
                (format!("200 OK\r\nContent-Length: {}", body.len()), body)
            });
            let api = mega::Client::builder()
                .origin(http_url(&origin)?)
                .https(true)
                .max_retries(2)
                .build(Client::new())?;
            let tmp = NamedTempFile::new()?;
            let result = download_mega_node(api, link.clone(), tmp.reopen()?, &Cell::new(0)).await;
            if corrupt {
                let error = result.unwrap_err();
                assert!(
                    matches!(
                        error.downcast_ref::<mega::Error>(),
                        Some(mega::Error::CondensedMacMismatch)
                    ),
                    "{error:#}"
                );
            } else {
                assert_eq!(result?.as_deref(), Some("fixture model.zip"));
                assert_eq!(std::fs::read(tmp.path())?, b"demo model sample");
            }
        }
        Ok(())
    }
}
