//! Dedicated dispatcher thread for serialized GPU submission per IMPLEMENTATION.md §4.
//! All inference calls are wrapped in std::panic::catch_unwind to isolate FFI panics.

use crate::types::{EngineError, InferenceEngine};
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use openjev_core::types::{Decision, Query};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tracing::{error, info};

pub struct BatchRequest {
    pub context: String,
    pub queries: Vec<Query>,
    pub response_tx: oneshot::Sender<Result<Vec<Decision>, EngineError>>,
}

pub struct DispatcherConfig {
    pub max_batch_size: usize,
    pub max_wait_ms: u64,
    pub max_queue_depth: usize,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 16,
            max_wait_ms: 8,
            max_queue_depth: 2048,
        }
    }
}

pub struct BatchDispatcher {
    tx: Sender<BatchRequest>,
    stopping: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl BatchDispatcher {
    pub fn new<E: InferenceEngine + 'static>(engine: Arc<E>, config: DispatcherConfig) -> Self {
        let (tx, rx) = bounded::<BatchRequest>(config.max_queue_depth);
        let stopping = Arc::new(AtomicBool::new(false));
        let stopping_clone = Arc::clone(&stopping);

        let handle = thread::Builder::new()
            .name("mlx_dispatcher_thread".to_string())
            .spawn(move || {
                Self::worker_loop(engine, rx, stopping_clone, config);
            })
            .expect("Failed to spawn mlx_dispatcher_thread");

        info!("BatchDispatcher thread started.");
        Self {
            tx,
            stopping,
            handle: Some(handle),
        }
    }

    pub fn submit(
        &self,
        context: String,
        queries: Vec<Query>,
    ) -> oneshot::Receiver<Result<Vec<Decision>, EngineError>> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let req = BatchRequest {
            context,
            queries,
            response_tx: resp_tx,
        };

        match self.tx.try_send(req) {
            Ok(()) => resp_rx,
            Err(TrySendError::Full(dropped)) => {
                let _ = dropped.response_tx.send(Err(EngineError::QueueFull(
                    "GPU command queue at capacity".to_string(),
                )));
                resp_rx
            }
            Err(TrySendError::Disconnected(dropped)) => {
                let _ = dropped.response_tx.send(Err(EngineError::ChannelClosed));
                resp_rx
            }
        }
    }

    pub fn stop(&mut self) {
        self.stopping.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
            info!("BatchDispatcher thread stopped.");
        }
    }

    fn worker_loop<E: InferenceEngine + 'static>(
        engine: Arc<E>,
        rx: Receiver<BatchRequest>,
        stopping: Arc<AtomicBool>,
        config: DispatcherConfig,
    ) {
        while !stopping.load(Ordering::Relaxed) {
            // Block until first item arrives
            let first_req = match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(req) => req,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            };

            let mut batch = vec![first_req];
            let deadline = Instant::now() + Duration::from_millis(config.max_wait_ms);

            // Accumulate up to max_batch_size before deadline expires
            while batch.len() < config.max_batch_size {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                match rx.recv_timeout(deadline - now) {
                    Ok(req) => batch.push(req),
                    Err(_) => break,
                }
            }

            // Execute batch forward pass wrapped in catch_unwind
            for req in batch {
                let engine_ref = Arc::clone(&engine);
                let ctx = req.context;
                let qs = req.queries;

                // FFI panic safety: catch_unwind protects the dispatcher thread
                let result =
                    std::panic::catch_unwind(AssertUnwindSafe(|| engine_ref.decide(&ctx, &qs)));

                let send_res = match result {
                    Ok(inference_res) => inference_res,
                    Err(panic_payload) => {
                        let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "Unknown FFI panic in Metal execution".to_string()
                        };
                        error!("Caught FFI panic during inference: {}", panic_msg);
                        Err(EngineError::FfiPanicCaught(panic_msg))
                    }
                };

                let _ = req.response_tx.send(send_res);
            }
        }
    }
}

impl Drop for BatchDispatcher {
    fn drop(&mut self) {
        self.stop();
    }
}
