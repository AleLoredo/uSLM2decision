//! OpenJEV Calibration library: scaling and empirical ECE measurement.

pub mod temperature;

pub use temperature::{compute_ece, scale_logit, scaled_sigmoid, scaled_softmax, sigmoid};
