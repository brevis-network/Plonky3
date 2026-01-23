#![no_std]

extern crate alloc;

mod data_traits;
pub mod dft;
mod extension;
mod mds;
mod monty_31;
mod poseidon2;
mod utils;
pub use data_traits::*;
pub use mds::*;
pub use monty_31::*;
pub use poseidon2::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
mod no_packing;
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
pub use no_packing::*;
