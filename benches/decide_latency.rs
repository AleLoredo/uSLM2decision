//! Criterion benchmarks for decision latency and mask generation per AGENTS.md.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use openjev_core::mask::{create_block_causal_mask, create_segment_position_ids};
use openjev_core::types::{ChoiceOption, Query};
use openjev_engine::{BatchDispatcher, DispatcherConfig, InferenceEngine, MockEngine};
use std::sync::Arc;

fn bench_mask_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mask_generation");
    for &seq_len in &[256usize, 512usize, 1024usize] {
        let s_len = seq_len - 32;
        let q_lens = vec![16, 16];

        group.bench_with_input(
            BenchmarkId::new("block_causal_mask", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(create_block_causal_mask(
                        black_box(s_len),
                        black_box(&q_lens),
                        true,
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("segment_position_ids", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    black_box(create_segment_position_ids(
                        black_box(s_len),
                        black_box(&q_lens),
                    ))
                });
            },
        );
    }
    group.finish();
}

fn bench_engine_decision_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_decide");
    let engine = MockEngine::new("qwen2.5-0.5b", "calib-canonical-v1");

    let context = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let queries = vec![
        Query::Bool {
            id: Some("q_bool".to_string()),
            criteria: "Is the fox jumping?".to_string(),
        },
        Query::Choice {
            id: Some("q_choice".to_string()),
            criteria: Some("Animal type".to_string()),
            options: vec![
                ChoiceOption {
                    id: "fox".to_string(),
                    description: "Wild Fox".to_string(),
                },
                ChoiceOption {
                    id: "dog".to_string(),
                    description: "Domestic Dog".to_string(),
                },
            ],
        },
        Query::Score {
            id: Some("q_score".to_string()),
            rubric: "Agility score 1-5".to_string(),
            min: 1,
            max: 5,
        },
    ];

    group.bench_function("mock_engine_3_queries", |b| {
        b.iter(|| {
            black_box(
                engine
                    .decide(black_box(&context), black_box(&queries))
                    .unwrap(),
            )
        });
    });

    group.finish();
}

fn bench_dispatcher_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatcher_throughput");
    let engine = Arc::new(MockEngine::new("qwen2.5-0.5b", "calib-canonical-v1"));
    let dispatcher = Arc::new(BatchDispatcher::new(
        Arc::clone(&engine),
        DispatcherConfig::default(),
    ));

    let rt = tokio::runtime::Runtime::new().unwrap();
    let context = "Context text for benchmarking micro-batch dispatcher.".to_string();
    let queries = vec![Query::Bool {
        id: Some("q1".to_string()),
        criteria: "Check condition".to_string(),
    }];

    group.bench_function("dispatcher_submit_and_resolve", |b| {
        b.iter(|| {
            let d = Arc::clone(&dispatcher);
            let ctx = context.clone();
            let qs = queries.clone();
            rt.block_on(async {
                let rx = d.submit(ctx, qs);
                let _ = rx.await.unwrap().unwrap();
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mask_generation,
    bench_engine_decision_latency,
    bench_dispatcher_throughput
);
criterion_main!(benches);
