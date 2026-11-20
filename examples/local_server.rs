//! Benchmark without any API key: starts a small OpenAI-compatible server
//! inside this process (streams a reply after a 120 ms "thinking" delay) and
//! measures it with llm-bench's runner, printing the per-model statistics.
//!
//! ```text
//! cargo run --example local_server
//! ```
//!
//! Swap `base_url` for `http://localhost:11434` to measure a real Ollama.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use llm_bench::runner::BenchRunner;
use llm_bench::stats::model_reports;
use llm_bench::types::{BenchConfig, ProviderConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Answers every request with a streamed reply: 120 ms to the first token,
/// then four more chunks 15 ms apart.
async fn start_mock() -> std::io::Result<String> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}", listener.local_addr()?);
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buf = vec![0u8; 16 * 1024];
                let mut req = Vec::new();
                while !String::from_utf8_lossy(&req).contains("}]") {
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => req.extend_from_slice(&buf[..n]),
                    }
                }
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")
                    .await;
                tokio::time::sleep(std::time::Duration::from_millis(120)).await;
                for word in ["Hello", " from", " a", " local", " server"] {
                    let chunk = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{word}\"}}}}]}}\n\n"
                    );
                    let _ = sock.write_all(chunk.as_bytes()).await;
                    let _ = sock.flush().await;
                    tokio::time::sleep(std::time::Duration::from_millis(15)).await;
                }
                let _ = sock
                    .write_all(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n")
                    .await;
                let _ = sock.shutdown().await;
            });
        }
    });
    Ok(url)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = start_mock().await?;
    let config = BenchConfig {
        prompts: vec!["Say hi".into(), "Name a colour".into()],
        runs_per_prompt: 5,
        concurrency: 4,
        providers: vec![ProviderConfig {
            name: "openai".into(),
            model: "local-mock".into(),
            api_key: String::new(),
            base_url,
            max_tokens: 32,
        }],
        stream: true,
        retries: 0,
        warmup: 1,
    };
    let (results, failures) = BenchRunner::new()?.run_detailed(&config, |_, _| {}).await?;
    for m in model_reports(&results, &failures) {
        println!("{} ({} ok, {} failed)", m.model, m.succeeded, m.failed);
        if let Some(t) = m.ttft_ms {
            println!(
                "  time to first token: p50 {} ms, p95 {} ms, mean {:.1} ms",
                t.p50, t.p95, t.mean
            );
        }
        if let Some(t) = m.total_ms {
            println!("  full response:       p50 {} ms, p95 {} ms", t.p50, t.p95);
        }
        if let Some(t) = m.tpot_ms {
            println!("  time per token:      mean {:.1} ms", t.mean);
        }
    }
    Ok(())
}
