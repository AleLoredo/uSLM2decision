//! OpenJEV API crate: Axum REST endpoints, middleware, and request dispatching.

pub mod middleware;
pub mod routes;

pub use middleware::{auth_middleware, trace_id_middleware, REQUEST_ID_HEADER};
pub use routes::{create_router, AppState};
