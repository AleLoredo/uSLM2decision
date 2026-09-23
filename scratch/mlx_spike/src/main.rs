use mlx_rs::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing mlx-rs initialization...");
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);
    let c = mlx_rs::ops::matmul(&a, &b)?;
    mlx_rs::transforms::eval([&c])?;
    println!("Matrix multiplication result:\n{:?}", c);
    Ok(())
}
