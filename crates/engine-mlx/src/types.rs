//! Engine traits, configuration, and errors.

use openjev_core::types::{Answer, Question, UsageInfo};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Engine warming up or not ready")]
    NotReady,

    #[error("Inference execution failure: {0}")]
    InferenceFailed(String),

    #[error("FFI panic caught during Metal GPU execution: {0}")]
    FfiPanicCaught(String),

    #[error("Queue capacity exceeded: {0}")]
    QueueFull(String),

    #[error("Channel communication error")]
    ChannelClosed,
}

pub trait InferenceEngine: Send + Sync {
    fn decide(
        &self,
        state: &str,
        questions: &HashMap<String, Question>,
    ) -> Result<(HashMap<String, Answer>, UsageInfo), EngineError>;
    fn model_version(&self) -> &str;
    fn calibration_version(&self) -> &str;
    fn is_ready(&self) -> bool;
}
