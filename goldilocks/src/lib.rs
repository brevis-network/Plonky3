//! The prime field known as Goldilocks, defined as `F_p` where `p = 2^64 - 2^32 + 1`.

#![no_std]

extern crate alloc;

mod extension;
mod goldilocks;
mod mds;
mod poseidon2;

pub use goldilocks::*;
pub use mds::*;
pub use poseidon2::*;
