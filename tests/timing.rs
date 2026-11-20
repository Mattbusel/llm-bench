//! Measurement accuracy: a local server that streams with known delays, and
//! a check that llm-bench reports those delays.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use llm_bench::runner::BenchRunner;
use llm_bench::stats::model_reports;
use llm_bench::types::{BenchConfig, ProviderConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const FIRST_TOKEN_DELAY: Duration = Duration::from_millis(300);
const DECODE_DELAY: Duration = Duration::from_millis(200);

/// Serves every request as an SSE stream: wait FIRST_TOKEN_DELAY, send one
/// chunk, wait DECODE_DELAY, send the rest and usage (11 completion tokens).
async fn streaming_server(hits: Arc<AtomicUsize>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            let hits = Arc::clone(&hits);
            tokio::spawn(async move {
                let mut buf = vec![0u8; 16 * 1024];
                let mut got = Vec::new();
                // Read headers and the JSON body (Content-Length).
                loop {
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    got.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&got);
                    if let Some(h) = text.find("\r\n\r\n") {
                        let len = text[..h]
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        if got.len() >= h + 4 + len {
                            break;
                        }
                    }
                }
                hits.fetch_add(1, Ordering::SeqCst);
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")
                    .await;
                tokio::time::sleep(FIRST_TOKEN_DELAY).await;
                let _ = sock
                    .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
                    .await;
                let _ = sock.flush().await;
                tokio::time::sleep(DECODE_DELAY).await;
                let _ = sock.write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\" world, ten more tokens\"}}]}\n\n").await;
                let _ = sock
                    .write_all(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":11}}\n\ndata: [DONE]\n\n")
                    .await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test(flavor = "multi_thread")]
async fn streamed_timings_match_the_server() {
    let hits = Arc::new(AtomicUsize::new(0));
    let base = streaming_server(Arc::clone(&hits)).await;
    let config = BenchConfig {
        prompts: vec!["hi".into()],
        runs_per_prompt: 3,
        concurrency: 3,
        providers: vec![ProviderConfig {
            name: "openai".into(),
            model: "local".into(),
            api_key: String::new(),
            base_url: base,
            max_tokens: 16,
        }],
        stream: true,
        retries: 0,
        warmup: 1,
    };
    let (results, failures) = BenchRunner::new()
        .unwrap()
        .run_detailed(&config, |_, _| {})
        .await
        .unwrap();
    assert!(failures.is_empty(), "{failures:?}");
    assert_eq!(results.len(), 3);
    // The warm-up request reached the server but is not in the results.
    assert_eq!(hits.load(Ordering::SeqCst), 4);

    let first = FIRST_TOKEN_DELAY.as_millis() as u64;
    let total = (FIRST_TOKEN_DELAY + DECODE_DELAY).as_millis() as u64;
    for r in &results {
        let ttft = r.ttft_ms.unwrap();
        // Lower bounds are exact (the server waits that long); upper bounds
        // allow 150 ms of local scheduling and connection set-up.
        assert!(ttft >= first && ttft < first + 150, "ttft {ttft}");
        assert!(
            r.total_ms >= total && r.total_ms < total + 150,
            "total {}",
            r.total_ms
        );
        assert_eq!(r.latency_ms, ttft);
        assert_eq!(r.completion_tokens, 11);
        // 10 tokens after the first, over (total - ttft) ms.
        let tpot = r.tpot_ms.unwrap();
        assert!((tpot - (r.total_ms - ttft) as f64 / 10.0).abs() < 1e-9);
        assert!((20.0 - 1e-9..35.0).contains(&tpot), "tpot {tpot}");
    }
    let report = &model_reports(&results, &failures)[0];
    let ttft = report.ttft_ms.unwrap();
    assert_eq!(ttft.count, 3);
    assert!(ttft.p50 >= first as f64);
}
