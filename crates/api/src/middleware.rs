use axum::{
    body::Body,
    http::{header, HeaderValue, Request, Response, StatusCode},
    middleware::Next,
};
use openjev_core::types::ErrorResponse;
use uuid::Uuid;

pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Injects a unique X-Request-ID header if not present.
pub async fn trace_id_middleware(mut req: Request<Body>, next: Next) -> Response<Body> {
    let req_id = match req.headers().get(REQUEST_ID_HEADER) {
        Some(v) => v.clone(),
        None => {
            let id = Uuid::new_v4().to_string();
            HeaderValue::from_str(&id).unwrap_or_else(|_| HeaderValue::from_static("unknown"))
        }
    };

    req.headers_mut().insert(REQUEST_ID_HEADER, req_id.clone());

    let mut res = next.run(req).await;
    res.headers_mut().insert(REQUEST_ID_HEADER, req_id);
    res
}

/// Bearer token authentication middleware per IMPLEMENTATION.md §8.
pub async fn auth_middleware(
    req: Request<Body>,
    next: Next,
    valid_tokens: Vec<String>,
) -> Response<Body> {
    // Exempt endpoints: /healthz, /readyz, /metrics
    let path = req.uri().path();
    if path == "/healthz" || path == "/readyz" || path == "/metrics" {
        return next.run(req).await;
    }

    if let Some(auth_header) = req.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                if valid_tokens.iter().any(|t| t == token) {
                    return next.run(req).await;
                }
            }
        }
    }

    let err = ErrorResponse {
        error: "Unauthorized".to_string(),
        detail: "Missing or invalid Bearer token".to_string(),
    };
    let json_body = serde_json::to_string(&err).unwrap_or_default();
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json_body))
        .unwrap()
}
