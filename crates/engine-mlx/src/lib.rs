//! OpenJEV Inference Engine library: engine traits, mock engine, and GPU dispatcher.

pub mod dispatcher;
pub mod mock;
pub mod types;

pub use dispatcher::{BatchDispatcher, DispatcherConfig};
pub use mock::MockEngine;
pub use types::{EngineError, InferenceEngine};
