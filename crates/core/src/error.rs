//! Typed errors for openjev-core library crate.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Tokenization error: {0}")]
    TokenizerError(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Empty sequence provided")]
    EmptySequence,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}
