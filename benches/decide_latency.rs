//! Criterion benchmarks for decision latency and mask generation per AGENTS.md.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use openjev_core::mask::{create_block_causal_mask, create_segment_position_ids};
use openjev_core::types::Question;
use openjev_engine::{BatchDispatcher, DispatcherConfig, InferenceEngine, MockEngine};
use std::collections::HashMap;
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

    let state = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let mut questions = HashMap::new();
    questions.insert(
        "q_noul".to_string(),
        Question::Noul {
            instructions: "Is the fox jumping?".to_string(),
            criteria: None,
        },
    );

    let mut choice_criteria = HashMap::new();
    choice_criteria.insert("fox".to_string(), "Wild Fox".to_string());
    choice_criteria.insert("dog".to_string(), "Domestic Dog".to_string());
    questions.insert(
        "q_choice".to_string(),
        Question::Choice {
            instructions: "Animal type".to_string(),
            criteria: choice_criteria,
        },
    );

    questions.insert(
        "q_score".to_string(),
        Question::Score {
            instructions: "Agility score".to_string(),
            criteria: vec![
                "Poor".to_string(),
                "Average".to_string(),
                "Good".to_string(),
                "Excellent".to_string(),
            ],
        },
    );

    group.bench_function("mock_engine_3_questions", |b| {
        b.iter(|| {
            black_box(
                engine
                    .decide(black_box(&state), black_box(&questions))
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
    let state = "State text for benchmarking micro-batch dispatcher.".to_string();
    let mut questions = HashMap::new();
    questions.insert(
        "q1".to_string(),
        Question::Noul {
            instructions: "Check condition".to_string(),
            criteria: None,
        },
    );

    group.bench_function("dispatcher_submit_and_resolve", |b| {
        b.iter(|| {
            let d = Arc::clone(&dispatcher);
            let s = state.clone();
            let qs = questions.clone();
            rt.block_on(async {
                let rx = d.submit(s, qs);
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
