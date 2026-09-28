//! OpenJEV Inference Engine library: engine traits, mock engine, real metal engine, and GPU dispatcher.

pub mod dispatcher;
pub mod heads;
pub mod loader;
pub mod mock;
pub mod real_engine;
pub mod types;

pub use dispatcher::{BatchDispatcher, DispatcherConfig};
pub use mock::MockEngine;
pub use real_engine::RealMetalEngine;
pub use types::{EngineError, InferenceEngine};
