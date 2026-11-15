//! End-to-end: the `llm-bench` binary against a local OpenAI-compatible mock
//! server. Covers a provider defined in `--config`, a built-in provider
//! (`groq:`) pointed at the mock by the config file, `--repeat`,
//! `--concurrency`, `--stream`, the error rate, configured prices, and the
//! Markdown and JSON outputs.

#![cfg(feature = "cli")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Time the mock waits before the first streamed token (and before a
/// non-streamed answer, which then waits `DECODE` more).
const FIRST_TOKEN: Duration = Duration::from_millis(80);
const DECODE: Duration = Duration::from_millis(40);

#[derive(Default)]
struct Seen {
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    requests: AtomicUsize,
    /// (path, authorization header, model) of every request.
    log: Mutex<Vec<(String, String, String)>>,
}

/// An OpenAI-compatible server: model `fail-model` gets HTTP 500, every
/// other model an answer with 1000 prompt and 500 completion tokens,
/// streamed when the request asks for it.
async fn mock_server(seen: Arc<Seen>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let seen = Arc::clone(&seen);
            tokio::spawn(async move {
                // Keep-alive: serve requests on this connection until it closes.
                let mut pending: Vec<u8> = Vec::new();
                let mut buf = vec![0u8; 16 * 1024];
                loop {
                    let (head, body) = loop {
                        if let Some(h) = find(&pending, b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&pending[..h]).to_string();
                            let len = header(&head, "content-length")
                                .and_then(|v| v.parse::<usize>().ok())
                                .unwrap_or(0);
                            if pending.len() >= h + 4 + len {
                                let body = pending[h + 4..h + 4 + len].to_vec();
                                pending.drain(..h + 4 + len);
                                break (head, body);
                            }
                        }
                        match sock.read(&mut buf).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => pending.extend_from_slice(&buf[..n]),
                        }
                    };
                    let now = seen.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    seen.max_in_flight.fetch_max(now, Ordering::SeqCst);
                    seen.requests.fetch_add(1, Ordering::SeqCst);

                    let path = head.split_whitespace().nth(1).unwrap_or("").to_owned();
                    let auth = header(&head, "authorization").unwrap_or_default();
                    let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                    let model = req["model"].as_str().unwrap_or("").to_owned();
                    let stream = req["stream"].as_bool().unwrap_or(false);
                    seen.log.lock().unwrap().push((path, auth, model.clone()));

                    let usage = r#"{"prompt_tokens":1000,"completion_tokens":500}"#;
                    let ok = if model == "fail-model" {
                        tokio::time::sleep(FIRST_TOKEN).await;
                        let body = r#"{"error":{"message":"boom"}}"#;
                        let resp = format!(
                            "HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                            body.len()
                        );
                        done(&seen);
                        sock.write_all(resp.as_bytes()).await.is_ok()
                    } else if stream {
                        let mut ok = sock
                            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
                            .await
                            .is_ok();
                        tokio::time::sleep(FIRST_TOKEN).await;
                        ok &= chunk(
                            &mut sock,
                            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
                        )
                        .await;
                        tokio::time::sleep(DECODE).await;
                        done(&seen);
                        ok &= chunk(
                            &mut sock,
                            "data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n",
                        )
                        .await;
                        ok &= chunk(
                            &mut sock,
                            &format!(
                                "data: {{\"choices\":[],\"usage\":{usage}}}\n\ndata: [DONE]\n\n"
                            ),
                        )
                        .await;
                        ok && sock.write_all(b"0\r\n\r\n").await.is_ok()
                    } else {
                        tokio::time::sleep(FIRST_TOKEN + DECODE).await;
                        let body = format!(
                            r#"{{"choices":[{{"message":{{"role":"assistant","content":"Hello there"}}}}],"usage":{usage}}}"#
                        );
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                            body.len()
                        );
                        done(&seen);
                        sock.write_all(resp.as_bytes()).await.is_ok()
                    };
                    let _ = sock.flush().await;
                    if !ok {
                        return;
                    }
                }
            });
        }
    });
    format!("http://{addr}")
}

/// A request stops counting as in flight just before its last bytes are
/// written, so the client can never start the next one while it still counts.
fn done(seen: &Seen) {
    seen.in_flight.fetch_sub(1, Ordering::SeqCst);
}

async fn chunk(sock: &mut tokio::net::TcpStream, data: &str) -> bool {
    let framed = format!("{:x}\r\n{data}\r\n", data.len());
    let ok = sock.write_all(framed.as_bytes()).await.is_ok();
    let _ = sock.flush().await;
    ok
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn header(head: &str, name: &str) -> Option<String> {
    head.lines().skip(1).find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim()
            .eq_ignore_ascii_case(name)
            .then(|| v.trim().to_owned())
    })
}

fn temp_path(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::SeqCst);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "llm_bench_e2e_{}_{nanos}_{n}_{name}",
        std::process::id()
    ))
}

/// A config file that defines `local:` and points `groq:` at the mock.
fn write_config(base: &str) -> PathBuf {
    let path = temp_path("config.toml");
    std::fs::write(
        &path,
        format!(
            r#"
[providers.local]
base_url = "{base}/v1"
api_key = "local-key"

[providers.groq]
base_url = "{base}/openai/v1"

# USD per 1M tokens: 1000 in + 500 out = $0.002 per request.
[prices."local:fast-model"]
input = 1.0
output = 2.0

[prices."llama-3.3-70b-versatile"]
input = 0.5
output = 1.0
"#
        ),
    )
    .unwrap();
    path
}

/// Run the binary off the async runtime so the mock keeps serving.
async fn llm_bench(args: Vec<String>) -> Output {
    tokio::task::spawn_blocking(move || {
        Command::new(env!("CARGO_BIN_EXE_llm-bench"))
            .args(&args)
            .env("GROQ_API_KEY", "groq-test-key")
            .env("NO_COLOR", "1")
            .env_remove("LLM_BENCH_CONFIG")
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn markdown_comparison_of_config_and_builtin_providers() {
    let seen = Arc::new(Seen::default());
    let base = mock_server(Arc::clone(&seen)).await;
    let cfg = write_config(&base);
    let out = llm_bench(strings(&[
        "run",
        "--config",
        cfg.to_str().unwrap(),
        "--models",
        "local:fast-model,local:fail-model,groq:llama-3.3-70b-versatile",
        "--prompts",
        "first prompt,second prompt",
        "--repeat",
        "3",
        "--concurrency",
        "4",
        "--stream",
        "--output",
        "markdown",
    ]))
    .await;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout:\n{stdout}\nstderr:\n{stderr}");

    // 3 models x 2 prompts x 3 repeats.
    assert_eq!(seen.requests.load(Ordering::SeqCst), 18);
    assert!(seen.max_in_flight.load(Ordering::SeqCst) <= 4);

    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines[0],
        "| Model | OK | Errors | TTFT p50 (ms) | Total p50 (ms) | Total p95 (ms) | Tok/s | Cost/req | Total cost |"
    );
    let row = |name: &str| -> Vec<String> {
        let l = lines
            .iter()
            .find(|l| l.starts_with(&format!("| {name} |")))
            .unwrap_or_else(|| panic!("no row for {name} in\n{stdout}"));
        l.trim_matches('|')
            .split('|')
            .map(|c| c.trim().to_owned())
            .collect()
    };
    let num = |s: &str| {
        s.parse::<f64>()
            .unwrap_or_else(|_| panic!("not a number: {s}"))
    };

    let fast = row("local:fast-model");
    assert_eq!(&fast[1..3], ["6/6", "0.0%"]);
    let (ttft, p50, p95) = (num(&fast[3]), num(&fast[4]), num(&fast[5]));
    let first = FIRST_TOKEN.as_millis() as f64;
    let total = (FIRST_TOKEN + DECODE).as_millis() as f64;
    assert!(ttft >= first && ttft < first + 150.0, "ttft {ttft}");
    assert!(
        p50 >= total && p50 <= p95 && p95 < total + 300.0,
        "p50 {p50} p95 {p95}"
    );
    assert!(num(&fast[6]) > 0.0);
    assert_eq!(&fast[7..9], ["$0.002000", "$0.012000"]);

    let failed = row("local:fail-model");
    assert_eq!(
        failed[1..].to_vec(),
        ["0/6", "100.0%", "-", "-", "-", "-", "-", "-"]
    );

    let groq = row("groq:llama-3.3-70b-versatile");
    assert_eq!(&groq[1..3], ["6/6", "0.0%"]);
    // (1000 x 0.5 + 500 x 1.0) / 1M.
    assert_eq!(groq[7], "$0.001000");

    assert!(
        stdout.contains("Cheapest: groq:llama-3.3-70b-versatile"),
        "{stdout}"
    );
    // Status lines and failure reasons go to stderr, not into the Markdown.
    assert!(!stdout.contains("Benchmarking"));
    assert!(stderr.contains("12 succeeded, 6 failed"), "{stderr}");
    assert!(stderr.contains("HTTP 500"), "{stderr}");

    // The groq: request went to the configured URL with GROQ_API_KEY, and
    // the config-defined provider sent its own key.
    let log = seen.log.lock().unwrap();
    assert!(log
        .iter()
        .any(|(path, auth, model)| path == "/openai/v1/chat/completions"
            && auth == "Bearer groq-test-key"
            && model == "llama-3.3-70b-versatile"));
    assert!(log
        .iter()
        .any(|(path, auth, model)| path == "/v1/chat/completions"
            && auth == "Bearer local-key"
            && model == "fast-model"));
    drop(log);
    let _ = std::fs::remove_file(&cfg);
}

#[tokio::test(flavor = "multi_thread")]
async fn json_output_markdown_file_and_summary_file() {
    let seen = Arc::new(Seen::default());
    let base = mock_server(Arc::clone(&seen)).await;
    let cfg = write_config(&base);
    let md = temp_path("table.md");
    let summary = temp_path("summary.json");
    let out = llm_bench(strings(&[
        "run",
        "--config",
        cfg.to_str().unwrap(),
        "--models",
        "local:fast-model,local:fail-model",
        "--prompts",
        "hi",
        "--repeat",
        "4",
        "--output",
        "json",
        "--markdown-file",
        md.to_str().unwrap(),
        "--summary-file",
        summary.to_str().unwrap(),
    ]))
    .await;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout:\n{stdout}\nstderr:\n{stderr}");

    // stdout is exactly the per-request JSON array (successes only).
    let results: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let results = results.as_array().unwrap();
    assert_eq!(results.len(), 4);
    for r in results {
        assert_eq!(r["provider"], "local");
        assert_eq!(r["cost_source"], "config");
        assert!((r["cost_usd"].as_f64().unwrap() - 0.002).abs() < 1e-12);
        assert!(r.get("ttft_ms").is_none(), "not streamed: {r}");
        assert!(r["total_ms"].as_u64().unwrap() >= (FIRST_TOKEN + DECODE).as_millis() as u64);
    }

    let reports: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary).unwrap()).unwrap();
    let by_model = |m: &str| {
        reports
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["model"] == m)
            .cloned()
            .unwrap()
    };
    assert_eq!(by_model("fail-model")["error_rate"], 1.0);
    assert_eq!(by_model("fast-model")["error_rate"], 0.0);
    assert!(
        (by_model("fast-model")["cost_per_request_usd"]
            .as_f64()
            .unwrap()
            - 0.002)
            .abs()
            < 1e-12
    );
    assert!(by_model("fast-model")["total_ms"]["p95"].as_f64().is_some());

    let table = std::fs::read_to_string(&md).unwrap();
    assert!(table.starts_with("| Model | OK | Errors |"), "{table}");
    assert!(
        table.contains("| local:fast-model | 4/4 | 0.0% | - |"),
        "{table}"
    );

    for p in [cfg, md, summary] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrency_caps_requests_in_flight() {
    for (concurrency, expect_max) in [(1usize, 1usize), (3, 3)] {
        let seen = Arc::new(Seen::default());
        let base = mock_server(Arc::clone(&seen)).await;
        let cfg = write_config(&base);
        let start = Instant::now();
        let out = llm_bench(strings(&[
            "run",
            "--config",
            cfg.to_str().unwrap(),
            "--models",
            "local:fast-model",
            "--prompts",
            "hi",
            "--repeat",
            "6",
            "--concurrency",
            &concurrency.to_string(),
            "--output",
            "json",
        ]))
        .await;
        let elapsed = start.elapsed();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(seen.requests.load(Ordering::SeqCst), 6);
        assert_eq!(
            seen.max_in_flight.load(Ordering::SeqCst),
            expect_max,
            "--concurrency {concurrency}"
        );
        // 6 requests of 120 ms each, `concurrency` at a time.
        let waves = 6u32.div_ceil(concurrency as u32);
        assert!(elapsed >= (FIRST_TOKEN + DECODE) * waves, "{elapsed:?}");
        let _ = std::fs::remove_file(&cfg);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn all_requests_failing_exits_with_status_1() {
    let seen = Arc::new(Seen::default());
    let base = mock_server(Arc::clone(&seen)).await;
    let cfg = write_config(&base);
    let out = llm_bench(strings(&[
        "run",
        "--config",
        cfg.to_str().unwrap(),
        "--models",
        "local:fail-model",
        "--prompts",
        "hi",
        "--repeat",
        "2",
    ]))
    .await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("every request failed"), "{stderr}");
    let _ = std::fs::remove_file(&cfg);
}

#[tokio::test(flavor = "multi_thread")]
async fn config_provider_named_like_a_model_family_uses_its_own_url() {
    // [providers.deepseek-local] and [providers.gpt-proxy] start like the
    // bare deepseek-* and gpt-* model names. With vendor keys set, requests
    // must still go to the configured URL, not to api.deepseek.com or
    // api.openai.com.
    let seen = Arc::new(Seen::default());
    let base = mock_server(Arc::clone(&seen)).await;
    let cfg = temp_path("lookalike.toml");
    std::fs::write(
        &cfg,
        format!(
            r#"
[providers.deepseek-local]
base_url = "{base}/v1"

[providers.gpt-proxy]
base_url = "{base}/proxy/v1"
api_key_env = "LLM_BENCH_E2E_PROXY_KEY"

[providers.needs-key]
base_url = "{base}/v1"
api_key_env = "LLM_BENCH_E2E_UNSET_KEY"
"#
        ),
    )
    .unwrap();
    let run = |models: &'static str| {
        let cfg = cfg.clone();
        tokio::task::spawn_blocking(move || {
            Command::new(env!("CARGO_BIN_EXE_llm-bench"))
                .args(["run", "--config", cfg.to_str().unwrap(), "--models", models])
                .args(["--prompts", "hi", "--repeat", "1", "--output", "markdown"])
                .env("DEEPSEEK_API_KEY", "vendor-deepseek-key")
                .env("OPENAI_API_KEY", "vendor-openai-key")
                .env("LLM_BENCH_E2E_PROXY_KEY", "proxy-key")
                .env_remove("LLM_BENCH_E2E_UNSET_KEY")
                .env_remove("LLM_BENCH_CONFIG")
                .env("NO_COLOR", "1")
                .output()
                .unwrap()
        })
    };

    let out = run("deepseek-local:foo,gpt-proxy:gpt-4o").await.unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("| deepseek-local:foo |"), "{stdout}");
    assert!(stdout.contains("| gpt-proxy:gpt-4o |"), "{stdout}");
    {
        let log = seen.log.lock().unwrap();
        assert_eq!(log.len(), 2, "{log:?}");
        assert!(log.iter().any(|(path, auth, model)| path == "/v1/chat/completions"
            && auth.is_empty()
            && model == "foo"));
        assert!(log.iter().any(|(path, auth, model)| path == "/proxy/v1/chat/completions"
            && auth == "Bearer proxy-key"
            && model == "gpt-4o"));
    }

    // api_key_env names an unset variable: a setup error, no requests sent.
    let out = run("needs-key:m").await.unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("LLM_BENCH_E2E_UNSET_KEY is not set"), "{stderr}");
    assert_eq!(seen.requests.load(Ordering::SeqCst), 2);
    let _ = std::fs::remove_file(&cfg);
}
