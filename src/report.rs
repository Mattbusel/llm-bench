//! # Module: Report
//!
//! ## Responsibility
//! Aggregate raw `BenchResult` records into `BenchSummary` statistics
//! (p50/p99 latency, throughput, cost) and render them to the terminal
//! or serialise them to JSON.
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

use colored::Colorize;
use tabled::settings::object::Columns;
use tabled::settings::{Alignment, Modify, Style};
use tabled::{Table, Tabled};

use crate::error::BenchError;
use crate::types::{BenchFailure, BenchResult, BenchSummary};

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
    let cheapest = live
        .iter()
        .min_by(|a, b| a.avg_cost_usd.total_cmp(&b.avg_cost_usd))?;
    Some(format!(
        "{} {} (p50 {} ms)   {} {} (${:.6} per request)",
        "Fastest:".green().bold(),
        fastest.model,
        fastest.p50_latency_ms,
        "Cheapest:".green().bold(),
        cheapest.model,
        cheapest.avg_cost_usd
    ))
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

/// Print a coloured ASCII summary table to stdout.
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
            SummaryRow {
                provider: s.provider.clone(),
                model: s.model.clone(),
                p50_ms: cell(format!("{}", s.p50_latency_ms)),
                p99_ms: cell(format!("{}", s.p99_latency_ms)),
                tokens_per_sec: cell(format!("{:.1}", s.avg_tokens_per_sec)),
                avg_cost: cell(format!("${:.6}", s.avg_cost_usd)),
                total_cost: cell(format!("${:.6}", s.total_cost_usd)),
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
        println!("{line}");
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

    //  print_table (smoke tests  -  checks no panic)

    #[test]
    fn test_print_table_empty_does_not_panic() {
        // Should print "No results to display." without panicking
        print_table(&[]);
    }

    #[test]
    fn test_print_table_with_data_does_not_panic() {
        let results = vec![make_result("openai", "gpt-4o-mini", 100, 0.001, 40.0)];
        let summaries = generate_summary(&results);
        print_table(&summaries);
    }

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
