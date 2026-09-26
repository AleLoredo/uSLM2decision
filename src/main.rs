//! OpenJEV Native Inference Engine Server binary entry point.
//! Conforms to SPEC.md §4, ARCHITECTURE.md, and IMPLEMENTATION.md §4 & §8.

use anyhow::{Context, Result};
use axum::middleware;
use openjev_api::{create_router, trace_id_middleware, AppState};
use openjev_engine::{BatchDispatcher, DispatcherConfig, MockEngine};
use serde::Deserialize;
use std::fs;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Deserialize)]
struct Config {
    server: ServerConfig,
    batching: BatchingConfig,
    model: ModelConfig,
}

#[derive(Debug, Deserialize)]
struct ServerConfig {
    #[serde(default = "default_host")]
    host: String,
    #[serde(default = "default_port")]
    port: u16,
    #[serde(default)]
    auth: AuthConfig,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8000
}

#[derive(Debug, Default, Deserialize)]
struct AuthConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    api_keys: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BatchingConfig {
    #[serde(default = "default_batch_size")]
    max_batch_size: usize,
    #[serde(default = "default_wait_ms")]
    max_wait_ms: u64,
    #[serde(default = "default_queue_depth")]
    max_queue_depth: usize,
}

fn default_batch_size() -> usize {
    16
}

fn default_wait_ms() -> u64 {
    8
}

fn default_queue_depth() -> usize {
    2048
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    #[serde(default = "default_model_name")]
    name: String,
    #[serde(default = "default_calibration_version")]
    calibration_version: String,
}

fn default_model_name() -> String {
    "qwen2.5-0.5b".to_string()
}

fn default_calibration_version() -> String {
    "calib-canonical-v1".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: default_host(),
                port: default_port(),
                auth: AuthConfig::default(),
            },
            batching: BatchingConfig {
                max_batch_size: default_batch_size(),
                max_wait_ms: default_wait_ms(),
                max_queue_depth: default_queue_depth(),
            },
            model: ModelConfig {
                name: default_model_name(),
                calibration_version: default_calibration_version(),
            },
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,openjev=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting OpenJEV Native Inference Engine...");

    let config_path = "config/engine.toml";
    let cfg: Config = if std::path::Path::new(config_path).exists() {
        let content = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read {}", config_path))?;
        toml::from_str(&content).with_context(|| format!("Failed to parse {}", config_path))?
    } else {
        warn!("Config file {} not found; using defaults.", config_path);
        Config::default()
    };

    info!(
        "Loaded configuration: model='{}', batch_size={}, max_wait_ms={}, max_queue_depth={}",
        cfg.model.name,
        cfg.batching.max_batch_size,
        cfg.batching.max_wait_ms,
        cfg.batching.max_queue_depth
    );

    // 1. Initialize inference engine
    let engine = Arc::new(MockEngine::new(
        &cfg.model.name,
        &cfg.model.calibration_version,
    ));

    // 2. Initialize dynamic micro-batching dispatcher
    let dispatcher_cfg = DispatcherConfig {
        max_batch_size: cfg.batching.max_batch_size,
        max_wait_ms: cfg.batching.max_wait_ms,
        max_queue_depth: cfg.batching.max_queue_depth,
    };
    let dispatcher = Arc::new(BatchDispatcher::new(Arc::clone(&engine), dispatcher_cfg));

    // 3. Setup Axum application
    let app_state = AppState {
        engine: Arc::clone(&engine) as Arc<dyn openjev_engine::InferenceEngine>,
        dispatcher: Arc::clone(&dispatcher),
    };

    let mut router = create_router(app_state).layer(middleware::from_fn(trace_id_middleware));

    if cfg.server.auth.enabled {
        let valid_keys = cfg.server.auth.api_keys.clone();
        info!(
            "Bearer token authentication enabled with {} keys.",
            valid_keys.len()
        );
        router = router.layer(middleware::from_fn(move |req, next| {
            openjev_api::auth_middleware(req, next, valid_keys.clone())
        }));
    }

    let addr: SocketAddr = format!("{}:{}", cfg.server.host, cfg.server.port)
        .parse()
        .with_context(|| {
            format!(
                "Invalid bind address {}:{}",
                cfg.server.host, cfg.server.port
            )
        })?;

    info!("OpenJEV API server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server execution failure")?;

    info!("OpenJEV Server cleanly terminated.");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received SIGINT (Ctrl+C). Initiating graceful shutdown..."),
        _ = terminate => info!("Received SIGTERM. Initiating graceful shutdown..."),
    }
}
