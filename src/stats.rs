//! Per-model statistics for the `--summary-file` report: count, mean,
//! standard deviation, min, max and the p50 / p90 / p95 / p99 percentiles of
//! latency, time to first token, total time and time per output token.
//!
//! Percentiles use the nearest-rank method (the value at rank
//! `ceil(p / 100 * n)`), which always returns a measured value; with few
//! samples, p99 is simply the slowest request.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{BenchFailure, BenchResult};

/// Distribution of one measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Stats {
    /// Number of samples.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (n - 1); 0 with one sample.
    pub stddev: f64,
    /// Smallest sample.
    pub min: f64,
    /// Median (nearest rank).
    pub p50: f64,
    /// 90th percentile (nearest rank).
    pub p90: f64,
    /// 95th percentile (nearest rank).
    pub p95: f64,
    /// 99th percentile (nearest rank).
    pub p99: f64,
    /// Largest sample.
    pub max: f64,
}

impl Stats {
    /// Statistics of `samples`, or `None` when there are none. Non-finite
    /// values are ignored.
    pub fn of(samples: impl IntoIterator<Item = f64>) -> Option<Stats> {
        let mut v: Vec<f64> = samples.into_iter().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let mean = v.iter().sum::<f64>() / n as f64;
        let stddev = if n > 1 {
            (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        } else {
            0.0
        };
        let rank = |p: usize| v[(p * n).div_ceil(100).saturating_sub(1).min(n - 1)];
        Some(Stats {
            count: n,
            mean,
            stddev,
            min: v[0],
            p50: rank(50),
            p90: rank(90),
            p95: rank(95),
            p99: rank(99),
            max: v[n - 1],
        })
    }
}

/// Everything measured for one provider and model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelReport {
    /// Provider name.
    pub provider: String,
    /// Model identifier.
    pub model: String,
    /// Requests that succeeded.
    pub succeeded: usize,
    /// Requests that failed.
    pub failed: usize,
    /// `latency_ms` (time to first token with `--stream`, else full response).
    pub latency_ms: Option<Stats>,
    /// Time to the full response.
    pub total_ms: Option<Stats>,
    /// Time to first token (streamed requests only).
    pub ttft_ms: Option<Stats>,
    /// Time per output token after the first (streamed requests only).
    pub tpot_ms: Option<Stats>,
    /// Completion tokens per second over the whole request.
    pub tokens_per_second: Option<Stats>,
    /// Sum of request costs in USD (0 when the model has no price).
    pub total_cost_usd: f64,
    /// Retries spent on the successful requests.
    pub retries: u32,
}

/// One [`ModelReport`] per provider and model, sorted by provider then model.
pub fn model_reports(results: &[BenchResult], failures: &[BenchFailure]) -> Vec<ModelReport> {
    let mut groups: BTreeMap<(String, String), (Vec<&BenchResult>, usize)> = BTreeMap::new();
    for r in results {
        groups
            .entry((r.provider.clone(), r.model.clone()))
            .or_default()
            .0
            .push(r);
    }
    for f in failures {
        groups
            .entry((f.provider.clone(), f.model.clone()))
            .or_default()
            .1 += 1;
    }
    groups
        .into_iter()
        .map(|((provider, model), (rs, failed))| ModelReport {
            provider,
            model,
            succeeded: rs.len(),
            failed,
            latency_ms: Stats::of(rs.iter().map(|r| r.latency_ms as f64)),
            total_ms: Stats::of(rs.iter().map(|r| r.total_ms as f64)),
            ttft_ms: Stats::of(rs.iter().filter_map(|r| r.ttft_ms.map(|t| t as f64))),
            tpot_ms: Stats::of(rs.iter().filter_map(|r| r.tpot_ms)),
            tokens_per_second: Stats::of(
                rs.iter()
                    .filter(|r| r.completion_tokens > 0)
                    .map(|r| r.tokens_per_second),
            ),
            total_cost_usd: rs.iter().map(|r| r.cost_usd).sum(),
            retries: rs.iter().map(|r| r.retries).sum(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn known_values() {
        let s = Stats::of((1..=100).map(f64::from)).unwrap();
        assert_eq!((s.min, s.p50, s.p90, s.p95, s.p99, s.max), (1.0, 50.0, 90.0, 95.0, 99.0, 100.0));
        assert!((s.mean - 50.5).abs() < 1e-12);
        // Sample standard deviation of 1..=100.
        assert!((s.stddev - 29.011_491_975_882_016).abs() < 1e-9);
        let one = Stats::of([7.0]).unwrap();
        assert_eq!((one.p50, one.p99, one.stddev), (7.0, 7.0, 0.0));
        assert!(Stats::of([]).is_none());
        assert!(Stats::of([f64::NAN]).is_none());
    }

    proptest! {
        #[test]
        fn percentiles_are_ordered_samples(v in prop::collection::vec(0.0f64..1e7, 1..300)) {
            let s = Stats::of(v.clone()).unwrap();
            prop_assert!(s.min <= s.p50 && s.p50 <= s.p90 && s.p90 <= s.p95 && s.p95 <= s.p99 && s.p99 <= s.max);
            for p in [s.p50, s.p90, s.p95, s.p99] {
                prop_assert!(v.contains(&p), "percentile {} is not a sample", p);
            }
            prop_assert!(s.mean >= s.min - 1e-6 && s.mean <= s.max + 1e-6);
            prop_assert!(s.stddev >= 0.0);
        }
    }
}
