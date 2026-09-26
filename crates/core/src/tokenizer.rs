//! Tokenization utilities using the tokenizers crate and fallback tokenization.

use crate::error::CoreError;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Clone)]
pub struct Tokenizer {
    inner: Option<HfTokenizer>,
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, CoreError> {
        let p = path.as_ref();
        if p.exists() {
            let tok = HfTokenizer::from_file(p).map_err(|e| {
                CoreError::TokenizerError(format!("Failed to load tokenizer: {}", e))
            })?;
            Ok(Self { inner: Some(tok) })
        } else {
            Ok(Self { inner: None })
        }
    }

    pub fn fallback() -> Self {
        Self { inner: None }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, CoreError> {
        if let Some(ref tok) = self.inner {
            let encoding = tok
                .encode(text, false)
                .map_err(|e| CoreError::TokenizerError(format!("Encoding error: {}", e)))?;
            let ids = encoding.get_ids().to_vec();
            if ids.is_empty() {
                Ok(vec![1])
            } else {
                Ok(ids)
            }
        } else {
            // Fallback word hashing tokenizer
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.is_empty() {
                Ok(vec![1])
            } else {
                let ids = words
                    .into_iter()
                    .map(|w| {
                        let mut hash: u32 = 5381;
                        for byte in w.bytes() {
                            hash = ((hash << 5).wrapping_add(hash)).wrapping_add(byte as u32);
                        }
                        (hash % 32000) + 1
                    })
                    .collect();
                Ok(ids)
            }
        }
    }
}
