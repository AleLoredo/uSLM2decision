//! API data structures conforming to SPEC.md §4.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideRequest {
    pub context: String,
    pub queries: Vec<Query>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Query {
    #[serde(rename = "bool")]
    Bool {
        #[serde(default)]
        id: Option<String>,
        criteria: String,
    },
    #[serde(rename = "choice")]
    Choice {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        criteria: Option<String>,
        options: Vec<ChoiceOption>,
    },
    #[serde(rename = "score")]
    Score {
        #[serde(default)]
        id: Option<String>,
        rubric: String,
        #[serde(default = "default_min_score")]
        min: i32,
        #[serde(default = "default_max_score")]
        max: i32,
    },
}

fn default_min_score() -> i32 {
    1
}

fn default_max_score() -> i32 {
    5
}

impl Query {
    pub fn id(&self) -> Option<&str> {
        match self {
            Query::Bool { id, .. } => id.as_deref(),
            Query::Choice { id, .. } => id.as_deref(),
            Query::Score { id, .. } => id.as_deref(),
        }
    }

    pub fn primitive_type(&self) -> &'static str {
        match self {
            Query::Bool { .. } => "bool",
            Query::Choice { .. } => "choice",
            Query::Score { .. } => "score",
        }
    }

    pub fn text_content(&self) -> String {
        match self {
            Query::Bool { criteria, .. } => criteria.clone(),
            Query::Choice {
                criteria, options, ..
            } => {
                let crit = criteria.as_deref().unwrap_or("");
                let opts = options
                    .iter()
                    .map(|o| o.description.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("{} {}", crit, opts).trim().to_string()
            }
            Query::Score { rubric, .. } => rubric.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceOption {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Decision {
    #[serde(rename = "bool")]
    Bool {
        id: String,
        value: bool,
        confidence: f32,
    },
    #[serde(rename = "choice")]
    Choice {
        id: String,
        selected: String,
        confidence: f32,
        distribution: HashMap<String, f32>,
    },
    #[serde(rename = "score")]
    Score {
        id: String,
        score: f32,
        discrete_level: i32,
        confidence: f32,
        variance: f32,
        distribution: Vec<f32>,
    },
}

impl Decision {
    pub fn id(&self) -> &str {
        match self {
            Decision::Bool { id, .. } => id,
            Decision::Choice { id, .. } => id,
            Decision::Score { id, .. } => id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideResponse {
    pub model_version: String,
    pub calibration_version: String,
    pub decisions: Vec<Decision>,
    pub metrics: RequestMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub detail: String,
}
