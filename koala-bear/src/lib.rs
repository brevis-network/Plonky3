#![no_std]

extern crate alloc;

mod extension;
mod koala_bear;
mod poseidon2;

pub use koala_bear::*;
pub use poseidon2::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;
