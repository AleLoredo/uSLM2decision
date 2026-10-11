//! OpenJEV Core library: data types, tokenization, and attention masking.

pub mod error;
pub mod mask;
pub mod tokenizer;
pub mod types;

pub use error::CoreError;
pub use mask::{create_block_causal_mask, create_segment_position_ids, pack_multi_request_masks};
pub use tokenizer::Tokenizer;
pub use types::{Answer, DecideRequest, DecideResponse, ErrorResponse, Question, UsageInfo};
