// Slides from Google. The webview can't fetch Google's links itself (CORS), so
// this downloads the PDF and hands the bytes back.
//
// Signed out: Google's public export links, so only files shared "anyone with
// the link"; anything else fails with SIGN_IN_NEEDED and the UI offers sign-in.
// Signed in (google_sign_in): the Drive API with the user's token, so private
// files work and google_list_files can browse Drive. Sign-in is the
// installed-app OAuth flow: the browser opens Google's consent page (Google
// refuses sign-in inside embedded webviews) and redirects back to a one-shot
// listener on 127.0.0.1, with PKCE. The user brings their own OAuth client
// (Google Cloud Console, type "Desktop app"); its ID and secret stay in
// google.json (0600) in the app data dir across sign-outs, next to the refresh
// token while signed in.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use tauri_plugin_opener::OpenerExt;

const MAX_BYTES: u64 = 200 << 20;
const SCOPE: &str = "https://www.googleapis.com/auth/drive.readonly";
const SIGN_IN_WAIT: Duration = Duration::from_secs(300);
const API: &str = "https://www.googleapis.com/drive/v3/files";
// The UI matches this exact error to open its sign-in dialog.
const SIGN_IN_NEEDED: &str = "SIGN_IN_NEEDED";

#[derive(Serialize, Deserialize)]
struct Saved {
    client_id: String,
    client_secret: String,
    #[serde(default)]
    refresh_token: Option<String>,
}

// The current access token and when it stops working; refreshed on demand.
static ACCESS: Mutex<Option<(String, Instant)>> = Mutex::new(None);
// Set by google_sign_in_cancel: stops waiting for the browser's redirect.
static CANCEL: AtomicBool = AtomicBool::new(false);

fn agent() -> ureq::Agent {
    // ureq defaults to rustls; only native-tls is compiled in (see Cargo.toml).
    let tls = ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build();
    ureq::Agent::config_builder().tls_config(tls).build().into()
}

fn json<T: serde::de::DeserializeOwned>(mut resp: ureq::http::Response<ureq::Body>) -> Result<T, String> {
    serde_json::from_reader(resp.body_mut().as_reader()).map_err(|e| e.to_string())
}

fn saved_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app.path().app_data_dir().map_err(|e| e.to_string())?.join("google.json"))
}

fn load(app: &AppHandle) -> Option<Saved> {
    serde_json::from_slice(&std::fs::read(saved_path(app).ok()?).ok()?).ok()
}

fn store(app: &AppHandle, saved: &Saved) -> Result<(), String> {
    let path = saved_path(app)?;
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600); // a refresh token is a password
    let mut f = opts.open(&path).map_err(|e| e.to_string())?;
    f.write_all(&serde_json::to_vec(saved).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn random(n: usize) -> Result<String, String> {
    let mut buf = vec![0u8; n];
    getrandom::fill(&mut buf).map_err(|e| e.to_string())?;
    Ok(B64.encode(buf))
}

#[derive(Deserialize)]
struct TokenReply {
    access_token: String,
    expires_in: u64,
    refresh_token: Option<String>,
}

fn remember(reply: &TokenReply) {
    // A minute early, so a token never expires mid-download.
    let until = Instant::now() + Duration::from_secs(reply.expires_in.saturating_sub(60));
    *ACCESS.lock().unwrap() = Some((reply.access_token.clone(), until));
}

fn signed_in(app: &AppHandle) -> Option<Saved> {
    load(app).filter(|s| s.refresh_token.is_some())
}

// A bearer header for the signed-in user, refreshing the token when it's stale.
fn bearer(app: &AppHandle) -> Result<String, String> {
    let mut saved = signed_in(app).ok_or(SIGN_IN_NEEDED)?;
    if let Some((token, until)) = ACCESS.lock().unwrap().clone() {
        if Instant::now() < until {
            return Ok(format!("Bearer {token}"));
        }
    }
    let refresh = saved.refresh_token.clone().unwrap_or_default();
    let reply: TokenReply = match agent().post("https://oauth2.googleapis.com/token").send_form([
        ("client_id", saved.client_id.as_str()),
        ("client_secret", saved.client_secret.as_str()),
        ("refresh_token", refresh.as_str()),
        ("grant_type", "refresh_token"),
    ]) {
        // Revoked, or expired (an OAuth app in "Testing" gets 7-day tokens).
        Err(ureq::Error::StatusCode(400 | 401)) => {
            saved.refresh_token = None;
            store(app, &saved)?;
            return Err(SIGN_IN_NEEDED.into());
        }
        r => r.map_err(|e| e.to_string()).and_then(json)?,
    };
    remember(&reply);
    Ok(format!("Bearer {}", reply.access_token))
}

fn read_pdf(mut resp: ureq::http::Response<ureq::Body>, not_pdf: &str) -> Result<tauri::ipc::Response, String> {
    let pdf = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|t| t.contains("pdf"));
    if !pdf {
        return Err(not_pdf.into());
    }
    let mut bytes = Vec::new();
    resp.body_mut().as_reader().take(MAX_BYTES).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    Ok(tauri::ipc::Response::new(bytes))
}

// kind: "presentation" | "document" | "spreadsheets" (Google files, exported)
// or "file" (anything uploaded to Drive, downloaded as-is: must be a PDF).
#[tauri::command]
pub async fn fetch_google_pdf(app: AppHandle, kind: String, id: String) -> Result<tauri::ipc::Response, String> {
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
        return Err("that doesn't look like a Google file link".into());
    }
    if !["presentation", "document", "spreadsheets", "file"].contains(&kind.as_str()) {
        return Err(format!("unknown Google file kind {kind}"));
    }
    tauri::async_runtime::spawn_blocking(move || {
        let agent = agent();
        if signed_in(&app).is_none() {
            let url = if kind == "file" {
                format!("https://drive.google.com/uc?export=download&id={id}")
            } else {
                format!("https://docs.google.com/{kind}/d/{id}/export?format=pdf")
            };
            // Private files get a 401/403/404 or a sign-in page instead of the PDF.
            let resp = agent.get(&url).call().map_err(|e| match e {
                ureq::Error::StatusCode(401 | 403 | 404) => SIGN_IN_NEEDED.to_string(),
                e => e.to_string(),
            })?;
            return read_pdf(resp, SIGN_IN_NEEDED);
        }
        let auth = bearer(&app)?;
        let not_visible = |e: ureq::Error| match e {
            ureq::Error::StatusCode(403 | 404) => "your Google account can't open that file".to_string(),
            e => e.to_string(),
        };
        let url = if kind == "file" {
            format!("{API}/{id}?alt=media&supportsAllDrives=true")
        } else {
            // The export link, not files.export: that endpoint caps at 10 MB,
            // and image-heavy decks go over.
            #[derive(Deserialize)]
            struct Links {
                #[serde(rename = "exportLinks", default)]
                export_links: std::collections::HashMap<String, String>,
            }
            let links: Links = agent
                .get(&format!("{API}/{id}?fields=exportLinks&supportsAllDrives=true"))
                .header("Authorization", &auth)
                .call()
                .map_err(not_visible)
                .and_then(json)?;
            links.export_links.get("application/pdf").cloned().ok_or("Google can't export that file as PDF")?
        };
        let resp = agent.get(&url).header("Authorization", &auth).call().map_err(not_visible)?;
        read_pdf(resp, "that Drive file isn't a PDF. Open it in Google Slides or Docs and paste that link")
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(Serialize)]
pub struct DriveFile {
    id: String,
    name: String,
    kind: &'static str,
    modified: String,
}

// The signed-in user's Slides, Docs and PDFs, most recently viewed first;
// `query` filters by name.
#[tauri::command]
pub async fn google_list_files(app: AppHandle, query: String) -> Result<Vec<DriveFile>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let auth = bearer(&app)?;
        let mut q = "trashed=false and (mimeType='application/vnd.google-apps.presentation' or mimeType='application/vnd.google-apps.document' or mimeType='application/pdf')".to_string();
        let name = query.trim();
        if !name.is_empty() {
            q += &format!(" and name contains '{}'", name.replace('\\', "\\\\").replace('\'', "\\'"));
        }
        let mut url = url::Url::parse(API).unwrap();
        url.query_pairs_mut()
            .append_pair("q", &q)
            .append_pair("orderBy", "viewedByMeTime desc,modifiedTime desc")
            .append_pair("pageSize", "50")
            .append_pair("fields", "files(id,name,mimeType,modifiedTime)")
            .append_pair("supportsAllDrives", "true")
            .append_pair("includeItemsFromAllDrives", "true");
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Item {
            id: String,
            name: String,
            mime_type: String,
            #[serde(default)]
            modified_time: String,
        }
        #[derive(Deserialize)]
        struct List {
            files: Vec<Item>,
        }
        let list: List = agent().get(url.as_str()).header("Authorization", &auth).call().map_err(|e| e.to_string()).and_then(json)?;
        Ok(list
            .files
            .into_iter()
            .map(|f| DriveFile {
                kind: match f.mime_type.as_str() {
                    "application/vnd.google-apps.presentation" => "presentation",
                    "application/vnd.google-apps.document" => "document",
                    _ => "file",
                },
                id: f.id,
                name: f.name,
                modified: f.modified_time,
            })
            .collect())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    signed_in: bool,
    has_client: bool,
}

#[tauri::command]
pub fn google_status(app: AppHandle) -> Status {
    let saved = load(&app);
    Status { signed_in: saved.as_ref().is_some_and(|s| s.refresh_token.is_some()), has_client: saved.is_some() }
}

// Forgets the token but keeps the OAuth client, so signing back in is one click.
#[tauri::command]
pub fn google_sign_out(app: AppHandle) -> Result<(), String> {
    *ACCESS.lock().unwrap() = None;
    match load(&app) {
        Some(mut saved) => {
            saved.refresh_token = None;
            store(&app, &saved)
        }
        None => Ok(()),
    }
}

#[tauri::command]
pub fn google_sign_in_cancel() {
    CANCEL.store(true, Ordering::Relaxed);
}

// Empty client_id/client_secret reuse the saved OAuth client.
#[tauri::command]
pub async fn google_sign_in(app: AppHandle, client_id: String, client_secret: String) -> Result<(), String> {
    let (mut client_id, mut client_secret) = (client_id.trim().to_string(), client_secret.trim().to_string());
    if client_id.is_empty() || client_secret.is_empty() {
        let saved = load(&app).ok_or("enter your OAuth client ID and secret first")?;
        (client_id, client_secret) = (saved.client_id, saved.client_secret);
    }
    CANCEL.store(false, Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
        let redirect = format!("http://127.0.0.1:{}", listener.local_addr().map_err(|e| e.to_string())?.port());
        let verifier = random(32)?;
        let state = random(16)?;
        let mut auth = url::Url::parse("https://accounts.google.com/o/oauth2/v2/auth").unwrap();
        auth.query_pairs_mut()
            .append_pair("client_id", &client_id)
            .append_pair("redirect_uri", &redirect)
            .append_pair("response_type", "code")
            .append_pair("scope", SCOPE)
            .append_pair("code_challenge", &B64.encode(Sha256::digest(verifier.as_bytes())))
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", &state)
            .append_pair("access_type", "offline")
            .append_pair("prompt", "consent"); // always hands back a refresh token
        app.opener().open_url(auth.as_str(), None::<&str>).map_err(|e| e.to_string())?;

        // Wait for the browser's redirect, skipping stray requests (favicon).
        // A bad client ID never redirects (Google shows its error page), so
        // the UI's Cancel ends the wait.
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let deadline = Instant::now() + SIGN_IN_WAIT;
        let code = loop {
            let mut conn = match listener.accept() {
                Ok((conn, _)) => conn,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if CANCEL.load(Ordering::Relaxed) {
                        return Err("sign-in cancelled".into());
                    }
                    if Instant::now() > deadline {
                        return Err("Google sign-in timed out".into());
                    }
                    std::thread::sleep(Duration::from_millis(200));
                    continue;
                }
                Err(e) => return Err(e.to_string()),
            };
            // A browser preconnect may never send a request; don't hang on it.
            let _ = conn.set_nonblocking(false);
            let _ = conn.set_read_timeout(Some(Duration::from_secs(5)));
            let mut line = String::new();
            if BufReader::new(&conn).read_line(&mut line).is_err() {
                continue;
            }
            let path = line.split_whitespace().nth(1).unwrap_or("/");
            let query: std::collections::HashMap<_, _> =
                url::Url::parse(&format!("http://localhost{path}")).map_err(|e| e.to_string())?.query_pairs().into_owned().collect();
            if query.get("state") != Some(&state) {
                let _ = conn.write_all(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n");
                continue;
            }
            let msg = match query.get("code") {
                Some(_) => "Signed in. You can close this tab and go back to Speech.",
                None => "Sign-in was cancelled. You can close this tab.",
            };
            let page = format!("<!doctype html><meta charset=utf-8><title>Speech</title><p style=\"font:16px system-ui;margin:3em\">{msg}</p>");
            let _ = conn.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n{page}").as_bytes());
            match query.get("code") {
                Some(code) => break code.clone(),
                None => return Err(format!("Google sign-in cancelled ({})", query.get("error").map_or("no code", |e| e.as_str()))),
            }
        };

        let reply: TokenReply = agent()
            .post("https://oauth2.googleapis.com/token")
            .send_form([
                ("code", code.as_str()),
                ("client_id", client_id.as_str()),
                ("client_secret", client_secret.as_str()),
                ("redirect_uri", redirect.as_str()),
                ("grant_type", "authorization_code"),
                ("code_verifier", verifier.as_str()),
            ])
            .map_err(|e| match e {
                ureq::Error::StatusCode(400 | 401) => "Google rejected the sign-in. Check the client ID and secret".to_string(),
                e => e.to_string(),
            })
            .and_then(json)?;
        let refresh_token = Some(reply.refresh_token.clone().ok_or("Google didn't return a refresh token")?);
        remember(&reply);
        store(&app, &Saved { client_id, client_secret, refresh_token })
    })
    .await
    .map_err(|e| e.to_string())?
}
