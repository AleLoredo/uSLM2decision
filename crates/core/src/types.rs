//! API data structures conforming to SPEC.md §4.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideRequest {
    pub context: String,
    #[serde(deserialize_with = "deserialize_queries")]
    pub queries: Vec<Query>,
}

fn deserialize_queries<'de, D>(deserializer: D) -> Result<Vec<Query>, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Value::deserialize(deserializer)?;
    match val {
        Value::Array(arr) => {
            serde_json::from_value(Value::Array(arr)).map_err(de::Error::custom)
        }
        Value::Object(map) => {
            let mut queries = Vec::new();
            for (id, mut q_val) in map {
                if let Value::Object(ref mut q_map) = q_val {
                    if !q_map.contains_key("id") {
                        q_map.insert("id".to_string(), Value::String(id));
                    }
                }
                let query: Query = serde_json::from_value(q_val).map_err(de::Error::custom)?;
                queries.push(query);
            }
            Ok(queries)
        }
        _ => Err(de::Error::custom("queries must be a list or a map of query objects")),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_decide_request_from_array() {
        let json_str = r#"{
            "context": "test context",
            "queries": [
                {"type": "bool", "id": "q1", "criteria": "is valid"}
            ]
        }"#;
        let req: DecideRequest = serde_json::from_str(json_str).expect("failed to parse array");
        assert_eq!(req.queries.len(), 1);
        assert_eq!(req.queries[0].id(), Some("q1"));
    }

    #[test]
    fn test_deserialize_decide_request_from_map() {
        let json_str = r#"{
            "context": "test context",
            "queries": {
                "q_bool": {
                    "type": "bool",
                    "criteria": "is valid"
                },
                "q_score": {
                    "type": "score",
                    "rubric": "1 to 5"
                }
            }
        }"#;
        let req: DecideRequest = serde_json::from_str(json_str).expect("failed to parse map");
        assert_eq!(req.queries.len(), 2);
    }
}
