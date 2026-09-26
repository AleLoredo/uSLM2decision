//! Axum REST endpoints implementing SPEC.md §4.

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use openjev_core::types::{DecideRequest, DecideResponse, ErrorResponse, RequestMetrics};
use openjev_engine::{BatchDispatcher, EngineError, InferenceEngine};
use std::sync::Arc;
use std::time::Instant;
use tracing::error;

pub struct AppState {
    pub engine: Arc<dyn InferenceEngine>,
    pub dispatcher: Arc<BatchDispatcher>,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/v1/decide", post(decide))
        .with_state(Arc::new(state))
}

pub async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

pub async fn readyz(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.engine.is_ready() {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ready",
                "model_version": state.engine.model_version()
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "ServiceUnavailable",
                "detail": "Engine is warming up"
            })),
        )
    }
}

pub async fn decide(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DecideRequest>,
) -> Response {
    if payload.queries.is_empty() {
        let err = ErrorResponse {
            error: "BadRequest".to_string(),
            detail: "Queries array cannot be empty".to_string(),
        };
        return (StatusCode::BAD_REQUEST, Json(err)).into_response();
    }

    let start = Instant::now();
    let rx = state.dispatcher.submit(payload.context, payload.queries);

    match rx.await {
        Ok(Ok(decisions)) => {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            let resp = DecideResponse {
                model_version: state.engine.model_version().to_string(),
                calibration_version: state.engine.calibration_version().to_string(),
                decisions,
                metrics: RequestMetrics { latency_ms },
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Ok(Err(EngineError::QueueFull(msg))) => {
            let mut headers = HeaderMap::new();
            headers.insert(header::RETRY_AFTER, "2".parse().unwrap());
            let err = ErrorResponse {
                error: "ServiceUnavailable".to_string(),
                detail: msg,
            };
            (StatusCode::SERVICE_UNAVAILABLE, headers, Json(err)).into_response()
        }
        Ok(Err(EngineError::NotReady)) => {
            let err = ErrorResponse {
                error: "ServiceUnavailable".to_string(),
                detail: "Engine is warming up".to_string(),
            };
            (StatusCode::SERVICE_UNAVAILABLE, Json(err)).into_response()
        }
        Ok(Err(EngineError::InferenceFailed(msg))) => {
            let err = ErrorResponse {
                error: "BadRequest".to_string(),
                detail: msg,
            };
            (StatusCode::BAD_REQUEST, Json(err)).into_response()
        }
        Ok(Err(EngineError::FfiPanicCaught(msg))) => {
            error!("Returning 500 due to caught FFI panic: {}", msg);
            let err = ErrorResponse {
                error: "InternalError".to_string(),
                detail: format!("Inference engine failure: {}", msg),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err)).into_response()
        }
        Ok(Err(err)) => {
            let err_resp = ErrorResponse {
                error: "InternalError".to_string(),
                detail: err.to_string(),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err_resp)).into_response()
        }
        Err(_) => {
            let err = ErrorResponse {
                error: "InternalError".to_string(),
                detail: "Inference worker canceled request".to_string(),
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(err)).into_response()
        }
    }
}
