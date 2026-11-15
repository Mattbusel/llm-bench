//! # Module: Report
//!
//! ## Responsibility
//! Render benchmark results: the provider comparison table (time to first
//! token, p50/p95 total latency, tokens per second, error rate, cost per
//! request) for the terminal and as Markdown, the per-request JSON and CSV,
//! and the older `BenchSummary` aggregation (p50/p99 latency, throughput,
//! cost) kept for library users.
//!
//! ## Guarantees
//! - `generate_summary` is deterministic for a given input slice.
//! - p50/p99 computation uses sorted-index method (no floating-point percentile
//!   interpolation).
//! - `print_table` never panics on an empty summary slice.
//! - This module never panics.
//!
//! ## NOT Responsible For
//! - API calls (see `providers.rs`)
//! - CLI argument parsing (see `cli.rs`)

#[cfg(feature = "cli")]
use colored::Colorize;
#[cfg(feature = "cli")]
use tabled::settings::object::Columns;
#[cfg(feature = "cli")]
use tabled::settings::{Alignment, Modify, Style};
#[cfg(feature = "cli")]
use tabled::{builder::Builder, Table, Tabled};

use crate::error::BenchError;
use crate::stats::{ModelReport, Stats};
use crate::types::{BenchFailure, BenchResult, BenchSummary};

//  Comparison table (0.4)

/// Column headers of the comparison table, in order.
pub const COMPARISON_HEADERS: [&str; 9] = [
    "Model",
    "OK",
    "Errors",
    "TTFT p50 (ms)",
    "Total p50 (ms)",
    "Total p95 (ms)",
    "Tok/s",
    "Cost/req",
    "Total cost",
];

/// One row of cells per model, in the order of [`COMPARISON_HEADERS`]:
/// `provider:model`, successful/attempted requests, error rate, time to
/// first token p50 (`-` unless streamed), total latency p50 and p95,
/// median tokens per second, mean cost per successful request and total
/// cost (`n/a` for a model without a price, `-` when every request failed).
///
/// # Panics
/// This function never panics.
pub fn comparison_rows(reports: &[ModelReport]) -> Vec<[String; 9]> {
    reports
        .iter()
        .map(|m| {
            let dead = m.succeeded == 0;
            let dash = || "-".to_owned();
            let ms = |s: Option<Stats>, f: fn(&Stats) -> f64| {
                s.filter(|_| !dead)
                    .map_or_else(dash, |s| format!("{:.0}", f(&s)))
            };
            let money = |v: f64| {
                if dead {
                    dash()
                } else if m.cost_per_request_usd.is_none() {
                    "n/a".to_owned()
                } else {
                    format!("${v:.6}")
                }
            };
            [
                format!("{}:{}", m.provider, m.model),
                format!("{}/{}", m.succeeded, m.succeeded + m.failed),
                format!("{:.1}%", m.error_rate * 100.0),
                ms(m.ttft_ms, |s| s.p50),
                ms(m.total_ms, |s| s.p50),
                ms(m.total_ms, |s| s.p95),
                m.tokens_per_second
                    .filter(|_| !dead)
                    .map_or_else(dash, |s| format!("{:.1}", s.p50)),
                money(m.cost_per_request_usd.unwrap_or(0.0)),
                money(m.total_cost_usd),
            ]
        })
        .collect()
}

/// The comparison table as a GitHub-flavoured Markdown table (numbers
/// right-aligned), followed by the verdict line when two or more models
/// produced results. Pipes in model names are escaped.
///
/// # Panics
/// This function never panics.
pub fn markdown_table(reports: &[ModelReport]) -> String {
    let esc = |c: &str| c.replace('|', "\\|");
    let mut out = String::new();
    out.push_str(&format!("| {} |\n", COMPARISON_HEADERS.join(" | ")));
    out.push_str("|:---|");
    out.push_str(&"---:|".repeat(COMPARISON_HEADERS.len() - 1));
    out.push('\n');
    for row in comparison_rows(reports) {
        let cells: Vec<String> = row.iter().map(|c| esc(c)).collect();
        out.push_str(&format!("| {} |\n", cells.join(" | ")));
    }
    if let Some(v) = comparison_verdict(reports) {
        out.push('\n');
        out.push_str(&v);
        out.push('\n');
    }
    out
}

/// "Fastest / Highest throughput / Cheapest" line under the comparison
/// table, or `None` when fewer than two models produced results. Fastest is
/// by time to first token when every model was streamed, else by total p50;
/// cheapest compares only models with a price.
///
/// # Panics
/// This function never panics.
pub fn comparison_verdict(reports: &[ModelReport]) -> Option<String> {
    let live: Vec<&ModelReport> = reports.iter().filter(|m| m.succeeded > 0).collect();
    if live.len() < 2 {
        return None;
    }
    let name = |m: &ModelReport| format!("{}:{}", m.provider, m.model);
    let all_ttft = live.iter().all(|m| m.ttft_ms.is_some());
    let (label, key): (&str, fn(&ModelReport) -> Option<f64>) = if all_ttft {
        ("TTFT p50", |m| m.ttft_ms.map(|s| s.p50))
    } else {
        ("total p50", |m| m.total_ms.map(|s| s.p50))
    };
    let mut parts = Vec::new();
    if let Some((m, v)) = live
        .iter()
        .filter_map(|m| key(m).map(|v| (*m, v)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
    {
        parts.push(format!("Fastest: {} ({label} {v:.0} ms)", name(m)));
    }
    if let Some((m, v)) = live
        .iter()
        .filter_map(|m| m.tokens_per_second.map(|s| (*m, s.p50)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
    {
        parts.push(format!("Highest throughput: {} ({v:.1} tok/s)", name(m)));
    }
    let priced: Vec<(&ModelReport, f64)> = live
        .iter()
        .filter_map(|m| m.cost_per_request_usd.map(|c| (*m, c)))
        .collect();
    if priced.len() >= 2 {
        if let Some((m, c)) = priced.iter().min_by(|a, b| a.1.total_cmp(&b.1)) {
            parts.push(format!("Cheapest: {} (${c:.6} per request)", name(m)));
        }
    }
    (!parts.is_empty()).then(|| parts.join("   "))
}

/// Print the comparison table to stdout (feature `cli`), with the verdict
/// line under it.
///
/// # Panics
/// This function never panics.
#[cfg(feature = "cli")]
pub fn print_comparison(reports: &[ModelReport]) {
    if reports.is_empty() {
        println!("{}", "No results to display.".yellow());
        return;
    }
    let mut b = Builder::default();
    b.push_record(COMPARISON_HEADERS);
    for row in comparison_rows(reports) {
        b.push_record(row);
    }
    let mut table = b.build();
    table
        .with(Style::rounded())
        .with(Modify::new(Columns::new(1..)).with(Alignment::right()));
    println!("\n{}", table.to_string().cyan());
    if let Some(line) = comparison_verdict(reports) {
        println!("{}", line.green());
    }
    if reports.iter().any(|m| m.succeeded > 0 && m.ttft_ms.is_none()) {
        println!("{}", "TTFT needs --stream.".dimmed());
    }
    println!();
}

//  Summary generation

/// Aggregate a flat list of results into per-provider-model summaries.
///
/// Groups results by `(provider, model)`, sorts latencies, and computes
/// p50/p99 via sorted-index selection.
///
/// # Panics
/// This function never panics.
pub fn generate_summary(results: &[BenchResult]) -> Vec<BenchSummary> {
    use std::collections::HashMap;

    // Group by (provider, model)
    let mut groups: HashMap<(String, String), Vec<&BenchResult>> = HashMap::new();
    for r in results {
        groups
            .entry((r.provider.clone(), r.model.clone()))
            .or_default()
            .push(r);
    }

    let mut summaries: Vec<BenchSummary> = groups
        .into_iter()
        .map(|((provider, model), group)| {
            let mut latencies: Vec<u64> = group.iter().map(|r| r.latency_ms).collect();
            latencies.sort_unstable();

            let n = latencies.len();
            let p50_latency_ms = percentile_sorted(&latencies, 50);
            let p99_latency_ms = percentile_sorted(&latencies, 99);

            let avg_tokens_per_sec = if n > 0 {
                group.iter().map(|r| r.tokens_per_second).sum::<f64>() / n as f64
            } else {
                0.0
            };

            let total_cost_usd: f64 = group.iter().map(|r| r.cost_usd).sum();
            let avg_cost_usd = if n > 0 {
                total_cost_usd / n as f64
            } else {
                0.0
            };

            BenchSummary {
                provider,
                model,
                p50_latency_ms,
                p99_latency_ms,
                avg_tokens_per_sec,
                avg_cost_usd,
                total_cost_usd,
                success_rate: 1.0, // all results in the slice succeeded
            }
        })
        .collect();

    // Stable sort by (provider, model) for deterministic output
    summaries.sort_by(|a, b| a.provider.cmp(&b.provider).then(a.model.cmp(&b.model)));

    summaries
}

/// Like [`generate_summary`], but folds in failed requests so each row's
/// `success_rate` is real (successes / attempts), and models whose every
/// request failed still get a row (with 0% success).
///
/// # Panics
/// This function never panics.
pub fn generate_summary_with_failures(
    results: &[BenchResult],
    failures: &[BenchFailure],
) -> Vec<BenchSummary> {
    use std::collections::HashMap;

    let mut failed: HashMap<(String, String), usize> = HashMap::new();
    for f in failures {
        *failed
            .entry((f.provider.clone(), f.model.clone()))
            .or_default() += 1;
    }
    let mut ok: HashMap<(String, String), usize> = HashMap::new();
    for r in results {
        *ok.entry((r.provider.clone(), r.model.clone())).or_default() += 1;
    }

    let mut summaries = generate_summary(results);
    for s in &mut summaries {
        let key = (s.provider.clone(), s.model.clone());
        let good = ok.get(&key).copied().unwrap_or(0);
        let bad = failed.get(&key).copied().unwrap_or(0);
        s.success_rate = if good + bad > 0 {
            good as f64 / (good + bad) as f64
        } else {
            0.0
        };
    }
    for (provider, model) in failed.into_keys() {
        if !ok.contains_key(&(provider.clone(), model.clone())) {
            summaries.push(BenchSummary {
                provider,
                model,
                p50_latency_ms: 0,
                p99_latency_ms: 0,
                avg_tokens_per_sec: 0.0,
                avg_cost_usd: 0.0,
                total_cost_usd: 0.0,
                success_rate: 0.0,
            });
        }
    }
    summaries.sort_by(|a, b| a.provider.cmp(&b.provider).then(a.model.cmp(&b.model)));
    summaries
}

/// One line per distinct failure reason, with a count and a hint on how to
/// fix the common ones. Returns an empty vector when nothing failed.
///
/// # Panics
/// This function never panics.
pub fn failure_lines(failures: &[BenchFailure]) -> Vec<String> {
    let mut seen: Vec<(String, usize)> = Vec::new();
    for f in failures {
        let mut msg = format!("{}/{}: {}", f.provider, f.model, f.error);
        if msg.len() > 220 {
            let mut cut = 220;
            while !msg.is_char_boundary(cut) {
                cut -= 1;
            }
            msg.truncate(cut);
            msg.push_str("...");
        }
        match seen.iter_mut().find(|(m, _)| *m == msg) {
            Some((_, n)) => *n += 1,
            None => seen.push((msg, 1)),
        }
    }
    seen.into_iter()
        .map(|(msg, n)| {
            let hint = failure_hint(&msg);
            let count = if n > 1 {
                format!("{n}x ")
            } else {
                String::new()
            };
            if hint.is_empty() {
                format!("{count}{msg}")
            } else {
                format!("{count}{msg}\n      hint: {hint}")
            }
        })
        .collect()
}

/// A short "how to fix it" for well-known failure messages.
fn failure_hint(msg: &str) -> &'static str {
    if msg.contains("HTTP 401") || msg.contains("HTTP 403") {
        "the API key was rejected; check OPENAI_API_KEY / ANTHROPIC_API_KEY"
    } else if msg.contains("HTTP 404") {
        "unknown model or wrong base URL; run `llm-bench models` or check --openai-base-url"
    } else if msg.contains("Rate limited") {
        "lower --concurrency or use fewer --runs"
    } else if msg.contains("error sending request") || msg.contains("transport error") {
        "could not reach the server; check your network or the base URL"
    } else {
        ""
    }
}

/// "Fastest: ... Cheapest: ..." line under the table, or `None` when fewer
/// than two models produced results (nothing to compare).
///
/// # Panics
/// This function never panics.
pub fn verdict_line(summaries: &[BenchSummary]) -> Option<String> {
    let live: Vec<&BenchSummary> = summaries.iter().filter(|s| s.success_rate > 0.0).collect();
    if live.len() < 2 {
        return None;
    }
    let fastest = live.iter().min_by_key(|s| s.p50_latency_ms)?;
    let mut line = format!(
        "{} {} (p50 {} ms)",
        "Fastest:",
        fastest.model,
        fastest.p50_latency_ms
    );
    // Only compare cost between models that have a price.
    let priced: Vec<&&BenchSummary> = live.iter().filter(|s| s.total_cost_usd > 0.0).collect();
    if priced.len() >= 2 {
        if let Some(cheapest) = priced
            .iter()
            .min_by(|a, b| a.avg_cost_usd.total_cmp(&b.avg_cost_usd))
        {
            line.push_str(&format!(
                "   {} {} (${:.6} per request)",
                "Cheapest:",
                cheapest.model,
                cheapest.avg_cost_usd
            ));
        }
    }
    Some(line)
}

/// Return the value at the given integer percentile from a **sorted** slice.
///
/// Uses the "nearest rank" method: index = ceil(p/100 * n) - 1, clamped.
///
/// # Panics
/// This function never panics (empty slice returns 0).
fn percentile_sorted(sorted: &[u64], p: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    // nearest-rank index
    let idx = (p * n).div_ceil(100).saturating_sub(1).min(n - 1);
    sorted[idx]
}

//  Table rendering

/// Row type for `tabled` table rendering.
#[cfg(feature = "cli")]
#[derive(Tabled)]
struct SummaryRow {
    #[tabled(rename = "Provider")]
    provider: String,
    #[tabled(rename = "Model")]
    model: String,
    #[tabled(rename = "P50 (ms)")]
    p50_ms: String,
    #[tabled(rename = "P99 (ms)")]
    p99_ms: String,
    #[tabled(rename = "Tok/s")]
    tokens_per_sec: String,
    #[tabled(rename = "Avg Cost")]
    avg_cost: String,
    #[tabled(rename = "Total Cost")]
    total_cost: String,
    #[tabled(rename = "Success")]
    success_rate: String,
}

/// Print a coloured ASCII summary table to stdout (feature `cli`).
#[cfg(feature = "cli")]
///
/// # Panics
/// This function never panics.
pub fn print_table(summaries: &[BenchSummary]) {
    if summaries.is_empty() {
        println!("{}", "No results to display.".yellow());
        return;
    }

    let rows: Vec<SummaryRow> = summaries
        .iter()
        .map(|s| {
            // A model whose every request failed has no numbers to show.
            let dead = s.success_rate == 0.0;
            let cell = |v: String| if dead { "-".to_owned() } else { v };
            // Unpriced local models report zero cost: say so instead of $0.
            let money = |v: f64| {
                if s.total_cost_usd == 0.0 {
                    cell("n/a".to_owned())
                } else {
                    cell(format!("${v:.6}"))
                }
            };
            SummaryRow {
                provider: s.provider.clone(),
                model: s.model.clone(),
                p50_ms: cell(format!("{}", s.p50_latency_ms)),
                p99_ms: cell(format!("{}", s.p99_latency_ms)),
                tokens_per_sec: cell(format!("{:.1}", s.avg_tokens_per_sec)),
                avg_cost: money(s.avg_cost_usd),
                total_cost: money(s.total_cost_usd),
                success_rate: format!("{:.0}%", s.success_rate * 100.0),
            }
        })
        .collect();

    let table = Table::new(rows)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(2..)).with(Alignment::right()))
        .to_string();
    println!("\n{}", table.cyan());
    if let Some(line) = verdict_line(summaries) {
        println!("{}", line.green());
    }
    println!();
}

//  JSON output

/// Serialise the full result set to a JSON string.
///
/// # Errors
/// Returns `BenchError::SerdeJson` if serialisation fails (should never happen
/// for well-formed `BenchResult` values).
///
/// # Panics
/// This function never panics.
pub fn print_results_json(results: &[BenchResult]) -> Result<String, BenchError> {
    serde_json::to_string_pretty(results).map_err(BenchError::SerdeJson)
}

//  CSV output

/// One CSV row per result, with a header, written with the `csv` crate so
/// prompts and answers containing commas, quotes or newlines stay intact.
///
/// Columns: provider, model, run_index, latency_ms, ttft_ms, total_ms,
/// prompt_tokens, completion_tokens, tokens_per_second, cost_usd, retries,
/// prompt, response_text. `ttft_ms` is empty for non-streamed requests.
///
/// # Errors
/// Returns `BenchError::Io` if the CSV writer fails.
pub fn results_csv(results: &[BenchResult]) -> Result<String, BenchError> {
    let mut w = csv::Writer::from_writer(Vec::new());
    let io = |e: csv::Error| BenchError::Io(std::io::Error::other(e));
    w.write_record([
        "provider",
        "model",
        "run_index",
        "latency_ms",
        "ttft_ms",
        "total_ms",
        "prompt_tokens",
        "completion_tokens",
        "tokens_per_second",
        "cost_usd",
        "retries",
        "prompt",
        "response_text",
    ])
    .map_err(io)?;
    for r in results {
        w.write_record([
            r.provider.clone(),
            r.model.clone(),
            r.run_index.to_string(),
            r.latency_ms.to_string(),
            r.ttft_ms.map(|t| t.to_string()).unwrap_or_default(),
            r.total_ms.to_string(),
            r.prompt_tokens.to_string(),
            r.completion_tokens.to_string(),
            format!("{:.3}", r.tokens_per_second),
            format!("{:.8}", r.cost_usd),
            r.retries.to_string(),
            r.prompt.clone(),
            r.response_text.clone(),
        ])
        .map_err(io)?;
    }
    let bytes = w
        .into_inner()
        .map_err(|e| BenchError::Io(std::io::Error::other(e.to_string())))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

//  Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BenchResult;

    //  Helpers

    fn make_result(
        provider: &str,
        model: &str,
        latency_ms: u64,
        cost: f64,
        tps: f64,
    ) -> BenchResult {
        BenchResult {
            provider: provider.into(),
            model: model.into(),
            prompt: "test prompt".into(),
            latency_ms,
            total_ms: latency_ms + 10,
            prompt_tokens: 10,
            completion_tokens: 20,
            cost_usd: cost,
            tokens_per_second: tps,
            response_text: "response".into(),
            run_index: 0,
            ttft_ms: None,
            retries: 0,
            tpot_ms: None,
            cached_prompt_tokens: 0,
            cache_write_prompt_tokens: 0,
            cost_source: None,
        }
    }

    //  failures

    fn fail(provider: &str, model: &str, err: &str) -> BenchFailure {
        BenchFailure {
            provider: provider.into(),
            model: model.into(),
            error: err.into(),
        }
    }

    #[test]
    fn test_summary_with_failures_real_success_rate() {
        let results = vec![
            make_result("openai", "gpt-4o-mini", 100, 0.001, 50.0),
            make_result("openai", "gpt-4o-mini", 200, 0.001, 50.0),
            make_result("openai", "gpt-4o-mini", 300, 0.001, 50.0),
        ];
        let failures = vec![fail("openai", "gpt-4o-mini", "HTTP 500")];
        let s = generate_summary_with_failures(&results, &failures);
        assert_eq!(s.len(), 1);
        assert!((s[0].success_rate - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_summary_with_failures_keeps_all_failed_model() {
        let results = vec![make_result("openai", "gpt-4o-mini", 100, 0.001, 50.0)];
        let failures = vec![fail("anthropic", "claude-haiku-4-5", "HTTP 401")];
        let s = generate_summary_with_failures(&results, &failures);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].provider, "anthropic");
        assert_eq!(s[0].success_rate, 0.0);
    }

    #[test]
    fn test_failure_lines_groups_and_hints() {
        let failures = vec![
            fail("openai", "gpt-4o", "API error: HTTP 401  -  bad key"),
            fail("openai", "gpt-4o", "API error: HTTP 401  -  bad key"),
        ];
        let lines = failure_lines(&failures);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("2x "));
        assert!(lines[0].contains("hint:"));
    }

    #[test]
    fn test_results_csv_quotes_commas_and_newlines() {
        let mut r = make_result("openai", "gpt-4o-mini", 120, 0.0001, 40.0);
        r.prompt = "Say \"hi\", then stop".into();
        r.response_text = "line one\nline two".into();
        r.ttft_ms = Some(35);
        let csv_text = results_csv(&[r]).unwrap();
        let mut rd = csv::Reader::from_reader(csv_text.as_bytes());
        let rows: Vec<csv::StringRecord> = rd.records().map(|x| x.unwrap()).collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(&rows[0][4], "35");
        assert_eq!(&rows[0][11], "Say \"hi\", then stop");
        assert_eq!(&rows[0][12], "line one\nline two");
    }

    #[test]
    fn test_verdict_line_needs_two_models() {
        let one = generate_summary(&[make_result("openai", "a", 100, 0.001, 50.0)]);
        assert!(verdict_line(&one).is_none());
        let two = generate_summary(&[
            make_result("openai", "a", 100, 0.002, 50.0),
            make_result("openai", "b", 300, 0.001, 50.0),
        ]);
        let line = verdict_line(&two).unwrap_or_default();
        assert!(line.contains("a (p50 100 ms)"), "got {line}");
        assert!(line.contains("b ($0.001000"), "got {line}");
    }

    #[test]
    fn test_verdict_line_skips_cheapest_for_unpriced_models() {
        let two = generate_summary(&[
            make_result("openai", "a", 100, 0.0, 50.0),
            make_result("openai", "b", 300, 0.0, 50.0),
        ]);
        let line = verdict_line(&two).unwrap_or_default();
        assert!(line.contains("a (p50 100 ms)"), "got {line}");
        assert!(!line.contains("Cheapest"), "got {line}");
    }

    //  percentile_sorted

    #[test]
    fn test_percentile_sorted_empty_returns_zero() {
        assert_eq!(percentile_sorted(&[], 50), 0);
    }

    #[test]
    fn test_percentile_sorted_single_element_any_percentile() {
        assert_eq!(percentile_sorted(&[42], 50), 42);
        assert_eq!(percentile_sorted(&[42], 99), 42);
        assert_eq!(percentile_sorted(&[42], 1), 42);
    }

    #[test]
    fn test_percentile_sorted_p50_of_even_count() {
        // [10, 20, 30, 40] → p50 nearest-rank: ceil(50/100*4)-1 = ceil(2)-1 = 1 → sorted[1]=20
        let data = vec![10u64, 20, 30, 40];
        assert_eq!(percentile_sorted(&data, 50), 20);
    }

    #[test]
    fn test_percentile_sorted_p99_of_100_elements() {
        let data: Vec<u64> = (1..=100).collect();
        // nearest-rank: ceil(99) - 1 = 98 → value 99
        assert_eq!(percentile_sorted(&data, 99), 99);
    }

    #[test]
    fn test_percentile_sorted_p100_returns_max() {
        let data = vec![5u64, 15, 25, 35, 45];
        assert_eq!(percentile_sorted(&data, 100), 45);
    }

    #[test]
    fn test_percentile_sorted_p1_returns_first() {
        let data = vec![100u64, 200, 300];
        assert_eq!(percentile_sorted(&data, 1), 100);
    }

    #[test]
    fn test_percentile_sorted_p50_of_two_elements() {
        let data = vec![100u64, 200];
        // nearest-rank: ceil(50/100*2)-1 = ceil(1)-1 = 0 → sorted[0]=100
        assert_eq!(percentile_sorted(&data, 50), 100);
    }

    #[test]
    fn test_percentile_sorted_p99_of_two_elements_returns_second() {
        let data = vec![100u64, 200];
        // nearest-rank: ceil(99/100*2)-1 = ceil(1.98)-1 = 2-1 = 1 → sorted[1]=200
        assert_eq!(percentile_sorted(&data, 99), 200);
    }

    #[test]
    fn test_percentile_sorted_all_same_values() {
        let data = vec![77u64; 10];
        assert_eq!(percentile_sorted(&data, 50), 77);
        assert_eq!(percentile_sorted(&data, 99), 77);
    }

    //  generate_summary

    #[test]
    fn test_generate_summary_empty_input_returns_empty() {
        let summaries = generate_summary(&[]);
        assert!(summaries.is_empty());
    }

    #[test]
    fn test_generate_summary_single_result() {
        let results = vec![make_result("openai", "gpt-4o-mini", 150, 0.001, 40.0)];
        let summaries = generate_summary(&results);
        assert_eq!(summaries.len(), 1);
        let s = &summaries[0];
        assert_eq!(s.provider, "openai");
        assert_eq!(s.model, "gpt-4o-mini");
        assert_eq!(s.p50_latency_ms, 150);
        assert_eq!(s.p99_latency_ms, 150);
        assert!((s.avg_tokens_per_sec - 40.0).abs() < 1e-9);
        assert!((s.avg_cost_usd - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_generate_summary_groups_by_provider_and_model() {
        let results = vec![
            make_result("openai", "gpt-4o-mini", 100, 0.001, 30.0),
            make_result("anthropic", "claude-haiku-4-5", 200, 0.002, 50.0),
        ];
        let summaries = generate_summary(&results);
        assert_eq!(summaries.len(), 2);
    }

    #[test]
    fn test_generate_summary_same_model_aggregated() {
        let results = vec![
            make_result("openai", "gpt-4o-mini", 100, 0.001, 30.0),
            make_result("openai", "gpt-4o-mini", 200, 0.002, 50.0),
            make_result("openai", "gpt-4o-mini", 300, 0.003, 70.0),
        ];
        let summaries = generate_summary(&results);
        assert_eq!(summaries.len(), 1, "same provider+model should be grouped");
    }

    #[test]
    fn test_generate_summary_p50_correct_for_three_values() {
        // sorted: [100, 200, 300] → p50 nearest-rank: ceil(1.5)-1=1 → 200
        let results = vec![
            make_result("openai", "gpt-4o", 300, 0.0, 0.0),
            make_result("openai", "gpt-4o", 100, 0.0, 0.0),
            make_result("openai", "gpt-4o", 200, 0.0, 0.0),
        ];
        let summaries = generate_summary(&results);
        assert_eq!(summaries[0].p50_latency_ms, 200);
    }

    #[test]
    fn test_generate_summary_p99_correct_for_100_values() {
        let results: Vec<BenchResult> = (1u64..=100)
            .map(|i| make_result("openai", "gpt-4o", i, 0.0, 0.0))
            .collect();
        let summaries = generate_summary(&results);
        assert_eq!(summaries[0].p99_latency_ms, 99);
    }

    #[test]
    fn test_generate_summary_total_cost_is_sum() {
        let results = vec![
            make_result("openai", "gpt-4o-mini", 100, 0.001, 0.0),
            make_result("openai", "gpt-4o-mini", 100, 0.002, 0.0),
            make_result("openai", "gpt-4o-mini", 100, 0.003, 0.0),
        ];
        let summaries = generate_summary(&results);
        assert!((summaries[0].total_cost_usd - 0.006).abs() < 1e-9);
    }

    #[test]
    fn test_generate_summary_avg_cost_is_mean() {
        let results = vec![
            make_result("openai", "gpt-4o-mini", 100, 0.001, 0.0),
            make_result("openai", "gpt-4o-mini", 100, 0.003, 0.0),
        ];
        let summaries = generate_summary(&results);
        assert!((summaries[0].avg_cost_usd - 0.002).abs() < 1e-9);
    }

    #[test]
    fn test_generate_summary_avg_tps_is_mean() {
        let results = vec![
            make_result("openai", "gpt-4o", 100, 0.0, 40.0),
            make_result("openai", "gpt-4o", 100, 0.0, 60.0),
        ];
        let summaries = generate_summary(&results);
        assert!((summaries[0].avg_tokens_per_sec - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_summary_success_rate_is_one() {
        let results = vec![make_result("openai", "gpt-4o-mini", 100, 0.001, 30.0)];
        let summaries = generate_summary(&results);
        assert!((summaries[0].success_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_generate_summary_is_sorted_by_provider_then_model() {
        let results = vec![
            make_result("openai", "gpt-4o", 100, 0.0, 0.0),
            make_result("anthropic", "claude-haiku-4-5", 100, 0.0, 0.0),
            make_result("openai", "gpt-4o-mini", 100, 0.0, 0.0),
        ];
        let summaries = generate_summary(&results);
        assert_eq!(summaries.len(), 3);
        // anthropic sorts before openai
        assert_eq!(summaries[0].provider, "anthropic");
        assert_eq!(summaries[1].provider, "openai");
        assert_eq!(summaries[2].provider, "openai");
        // gpt-4o sorts before gpt-4o-mini
        assert_eq!(summaries[1].model, "gpt-4o");
        assert_eq!(summaries[2].model, "gpt-4o-mini");
    }

    //  print_results_json

    #[test]
    fn test_print_results_json_empty_slice_returns_empty_array() {
        let json = print_results_json(&[]);
        assert!(json.is_ok());
        assert_eq!(json.unwrap_or_default().trim(), "[]");
    }

    #[test]
    fn test_print_results_json_contains_provider() {
        let results = vec![make_result("openai", "gpt-4o-mini", 100, 0.001, 40.0)];
        let json = print_results_json(&results);
        assert!(json.is_ok());
        let s = json.unwrap_or_default();
        assert!(s.contains("openai"), "JSON should contain provider");
    }

    #[test]
    fn test_print_results_json_is_valid_json() {
        let results = vec![
            make_result("openai", "gpt-4o-mini", 100, 0.001, 40.0),
            make_result("anthropic", "claude-haiku-4-5", 200, 0.002, 50.0),
        ];
        let json = print_results_json(&results).unwrap_or_default();
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok(), "output should be valid JSON");
    }

    #[test]
    fn test_print_results_json_array_length_matches_input() {
        let results = vec![
            make_result("openai", "gpt-4o", 100, 0.01, 30.0),
            make_result("openai", "gpt-4o", 120, 0.01, 32.0),
            make_result("anthropic", "claude-haiku-4-5", 90, 0.005, 55.0),
        ];
        let json = print_results_json(&results).unwrap_or_default();
        let arr: serde_json::Value = serde_json::from_str(&json).unwrap_or_default();
        let arr_len = arr.as_array().map(|a| a.len()).unwrap_or(0);
        assert_eq!(arr_len, 3);
    }

    //  comparison table

    fn streamed(provider: &str, model: &str, ttft: u64, total: u64, cost: Option<f64>) -> BenchResult {
        let mut r = make_result(provider, model, ttft, cost.unwrap_or(0.0), 100.0);
        r.ttft_ms = Some(ttft);
        r.total_ms = total;
        r.cost_source = cost.map(|_| crate::types::CostSource::LlmPrices);
        r
    }

    #[test]
    fn test_comparison_rows_and_markdown() {
        let results = vec![
            streamed("groq", "llama|70b", 100, 400, None),
            streamed("groq", "llama|70b", 120, 500, None),
            streamed("openai", "gpt-4o-mini", 300, 900, Some(0.0002)),
            streamed("openai", "gpt-4o-mini", 340, 1000, Some(0.0004)),
        ];
        let failures = vec![
            fail("openai", "gpt-4o-mini", "HTTP 500"),
            fail("together", "dead", "HTTP 401"),
        ];
        let reports = crate::stats::model_reports(&results, &failures);
        let rows = comparison_rows(&reports);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], [
            "groq:llama|70b", "2/2", "0.0%", "100", "400", "500", "100.0", "n/a", "n/a",
        ].map(String::from));
        assert_eq!(rows[1], [
            "openai:gpt-4o-mini", "2/3", "33.3%", "300", "900", "1000", "100.0", "$0.000300", "$0.000600",
        ].map(String::from));
        assert_eq!(rows[2], [
            "together:dead", "0/1", "100.0%", "-", "-", "-", "-", "-", "-",
        ].map(String::from));

        let md = markdown_table(&reports);
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[0], "| Model | OK | Errors | TTFT p50 (ms) | Total p50 (ms) | Total p95 (ms) | Tok/s | Cost/req | Total cost |");
        assert_eq!(lines[1], "|:---|---:|---:|---:|---:|---:|---:|---:|---:|");
        assert!(lines[2].starts_with("| groq:llama\\|70b | 2/2 |"), "{}", lines[2]);
        // Every table row has the same number of unescaped cell separators.
        for l in &lines[..5] {
            assert_eq!(l.replace("\\|", "").matches('|').count(), 10, "{l}");
        }
        assert_eq!(lines[5], "");
        assert!(lines[6].contains("Fastest: groq:llama|70b (TTFT p50 100 ms)"), "{}", lines[6]);
        // Only one model is priced, so no "Cheapest".
        assert!(!lines[6].contains("Cheapest"), "{}", lines[6]);
    }

    #[test]
    fn test_comparison_verdict_uses_total_without_stream_and_cheapest() {
        let mut a = make_result("openai", "a", 900, 0.002, 50.0);
        a.cost_source = Some(crate::types::CostSource::LlmPrices);
        let mut b = make_result("openai", "b", 300, 0.001, 80.0);
        b.cost_source = Some(crate::types::CostSource::Config);
        let reports = crate::stats::model_reports(&[a, b], &[]);
        let v = comparison_verdict(&reports).unwrap();
        assert!(v.contains("Fastest: openai:b (total p50 310 ms)"), "{v}");
        assert!(v.contains("Highest throughput: openai:b (80.0 tok/s)"), "{v}");
        assert!(v.contains("Cheapest: openai:b ($0.001000 per request)"), "{v}");
        assert!(comparison_verdict(&reports[..1]).is_none());
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_print_comparison_does_not_panic() {
        print_comparison(&[]);
        let reports = crate::stats::model_reports(&[make_result("openai", "m", 1, 0.0, 1.0)], &[]);
        print_comparison(&reports);
    }

    //  print_table (smoke tests  -  checks no panic)

    #[cfg(feature = "cli")]
    #[test]
    fn test_print_table_empty_does_not_panic() {
        // Should print "No results to display." without panicking
        print_table(&[]);
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_print_table_with_data_does_not_panic() {
        let results = vec![make_result("openai", "gpt-4o-mini", 100, 0.001, 40.0)];
        let summaries = generate_summary(&results);
        print_table(&summaries);
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_print_table_multiple_rows_does_not_panic() {
        let results = vec![
            make_result("openai", "gpt-4o", 100, 0.01, 30.0),
            make_result("anthropic", "claude-haiku-4-5", 200, 0.005, 50.0),
        ];
        let summaries = generate_summary(&results);
        print_table(&summaries);
    }
}
