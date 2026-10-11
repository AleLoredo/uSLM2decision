//! API data structures conforming to SPEC.md §4 and ARCHITECTURE.md.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideRequest {
    pub state: Value,
    #[serde(default)]
    pub model: Option<String>,
    pub questions: HashMap<String, Question>,
}

impl DecideRequest {
    /// Serializes state to canonical text representation for tokenization.
    pub fn serialize_state(&self) -> String {
        match &self.state {
            Value::String(s) => s.clone(),
            other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Question {
    #[serde(rename = "choice")]
    Choice {
        instructions: String,
        criteria: HashMap<String, String>,
    },
    #[serde(rename = "score")]
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    #[serde(rename = "noul")]
    Noul {
        instructions: String,
        #[serde(default)]
        criteria: Option<String>,
    },
}

impl Question {
    pub fn primitive_type(&self) -> &'static str {
        match self {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        }
    }

    pub fn text_content(&self) -> String {
        match self {
            Question::Choice { instructions, .. } => instructions.clone(),
            Question::Score { instructions, .. } => instructions.clone(),
            Question::Noul {
                instructions,
                criteria,
            } => {
                if let Some(ref c) = criteria {
                    format!("{} {}", instructions, c).trim().to_string()
                } else {
                    instructions.clone()
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Answer {
    #[serde(rename = "choice")]
    Choice {
        choice: String,
        confidence: f32,
        probabilities: HashMap<String, f32>,
    },
    #[serde(rename = "score")]
    Score {
        score: f32,
        confidence: f32,
        legend: HashMap<String, String>,
        probabilities: HashMap<String, f32>,
    },
    #[serde(rename = "noul")]
    Noul { noul: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageInfo {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideResponse {
    pub model: String,
    pub answers: HashMap<String, Answer>,
    pub usage: UsageInfo,
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
    fn test_deserialize_canonical_decide_request() {
        let json_str = r#"{
            "state": "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
            "model": "jev-latest",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this",
                    "criteria": {
                        "billing": "Payment or subscription issues",
                        "technical": "Bugs or integration problems",
                        "sales": "Pricing or account questions"
                    }
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated the customer appears",
                    "criteria": [
                        "Calm, just stating facts",
                        "Frustrated but civil",
                        "Very angry, strong language"
                    ]
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": "The message conveys urgency or time-sensitivity"
                }
            }
        }"#;

        let req: DecideRequest =
            serde_json::from_str(json_str).expect("failed to parse canonical request");
        assert_eq!(req.questions.len(), 3);
        assert_eq!(req.model.as_deref(), Some("jev-latest"));

        let q_dept = req
            .questions
            .get("department")
            .expect("department not found");
        assert_eq!(q_dept.primitive_type(), "choice");

        let q_frust = req
            .questions
            .get("frustration")
            .expect("frustration not found");
        assert_eq!(q_frust.primitive_type(), "score");

        let q_urgent = req.questions.get("is_urgent").expect("is_urgent not found");
        assert_eq!(q_urgent.primitive_type(), "noul");
    }

    #[test]
    fn test_deserialize_structured_json_state() {
        let json_str = r#"{
            "state": {
                "user_id": 1234,
                "msg": "cannot connect stripe"
            },
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "urgent message"
                }
            }
        }"#;

        let req: DecideRequest =
            serde_json::from_str(json_str).expect("failed to parse structured state");
        assert_eq!(req.questions.len(), 1);
        let serialized = req.serialize_state();
        assert!(serialized.contains("user_id"));
        assert!(serialized.contains("1234"));
    }
}
