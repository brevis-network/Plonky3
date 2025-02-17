use core::cmp::Reverse;
use std::array;
use std::marker::PhantomData;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_commit::ExtensionMmcs;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, FieldAlgebra};
use p3_fri::{prover, verifier, FriConfig, FriProof, TwoAdicFriGenericConfig};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_strict_usize;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyFriConfig = FriConfig<ChallengeMmcs>;

fn get_ldt_for_testing<R: Rng>(
    rng: &mut R,
    log_blowup: usize,
    log_arity: usize,
) -> (Perm, MyFriConfig) {
    let perm = Perm::new_from_rng_128(rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
    let fri_config = FriConfig {
        log_blowup,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs,
        log_arity,
        log_final_poly_len: 1,
    };
    (perm, fri_config)
}

fn fri_setup<const N: usize>(
    test_shapes: [i32; N],
    log_blowup: usize,
    log_arity: usize,
) -> (
    MyFriConfig,
    Vec<Vec<Challenge>>,
    (Challenger, Challenger),
    usize,
) {
    let rng = &mut ChaCha20Rng::seed_from_u64(0);
    let (perm, fc) = get_ldt_for_testing(rng, log_blowup, log_arity);
    let dft = Radix2Dit::default();

    let shift = Val::GENERATOR;

    let ldes: Vec<RowMajorMatrix<Val>> = test_shapes
        .iter()
        .map(|deg_bits| {
            let evals = RowMajorMatrix::<Val>::rand_nonzero(rng, 1 << deg_bits, 1);
            let mut lde = dft.coset_lde_batch(evals, log_blowup, shift);
            reverse_matrix_index_bits(&mut lde);
            lde
        })
        .collect();

    // Prover world
    let mut p_challenger = Challenger::new(perm.clone());
    let alpha: Challenge = p_challenger.sample_ext_element();

    let input: [_; 32] = core::array::from_fn(|log_height| {
        let matrices_with_log_height: Vec<&RowMajorMatrix<Val>> = ldes
            .iter()
            .filter(|m| log2_strict_usize(m.height()) == log_height)
            .collect();
        if matrices_with_log_height.is_empty() {
            None
        } else {
            let reduced: Vec<Challenge> = (0..(1 << log_height))
                .map(|r| {
                    alpha
                        .powers()
                        .zip(matrices_with_log_height.iter().flat_map(|m| m.row(r)))
                        .map(|(alpha_pow, v)| alpha_pow * v)
                        .sum()
                })
                .collect();
            Some(reduced)
        }
    });

    let input: Vec<Vec<Challenge>> = input.into_iter().rev().flatten().collect();

    let log_max_height = log2_strict_usize(input[0].len());

    let mut v_challenger = Challenger::new(perm);
    let _alpha: Challenge = v_challenger.sample_ext_element();

    (fc, input, (p_challenger, v_challenger), log_max_height)
}

fn fri_prove(
    fc: &MyFriConfig,
    input: Vec<Vec<Challenge>>,
    mut chal: Challenger,
    log_max_height: usize,
) -> FriProof<
    Challenge,
    ChallengeMmcs,
    <Challenger as GrindingChallenger>::Witness,
    Vec<(usize, Challenge)>,
> {
    prover::prove(
        &TwoAdicFriGenericConfig::<Vec<(usize, Challenge)>, ()>(PhantomData),
        fc,
        input.clone(),
        &mut chal,
        |idx| {
            // As our "input opening proof", just pass through the literal reduced openings.
            let mut ro = vec![];
            for v in &input {
                let log_height = log2_strict_usize(v.len());
                ro.push((log_height, v[idx >> (log_max_height - log_height)]));
            }
            ro.sort_by_key(|(lh, _)| Reverse(*lh));
            ro
        },
    )
}

fn bench_fri(c: &mut Criterion) {
    // FRI is kind of flaky depending on indexing luck
    let mut group = c.benchmark_group("fri");
    group.sample_size(10);
    let test_shape_start = 20;
    const TEST_SHAPE_LEN: usize = 5;
    let log_blowup = 1;

    for log_step_by in 0..3 {
        let mut test_shapes: [_; TEST_SHAPE_LEN] =
            array::from_fn(|step| test_shape_start - (step << log_step_by) as i32);
        test_shapes.reverse();
        for log_arity in [1usize, 2, 4] {
            group.bench_function(
                format!(
                    "prove/test_shapes={:?}/log_arity={}",
                    test_shapes, log_arity
                ),
                |b| {
                    b.iter_with_setup(
                        || fri_setup(test_shapes, log_blowup, log_arity),
                        |(fc, input, (p_challenger, _), log_max_height)| {
                            let _ = black_box(fri_prove(&fc, input, p_challenger, log_max_height));
                        },
                    )
                },
            );
        }
    }

    for log_step_by in 0..3 {
        let mut test_shapes: [_; TEST_SHAPE_LEN] =
            array::from_fn(|step| test_shape_start - (step << log_step_by) as i32);
        test_shapes.reverse();
        for log_arity in [1usize, 2, 4] {
            group.bench_function(
                format!(
                    "verify/test_shapes={:?}/log_arity={}",
                    test_shapes, log_arity
                ),
                |b| {
                    b.iter_with_setup(
                        || {
                            let (fc, input, (p_challenger, v_challenger), log_max_height) =
                                fri_setup(test_shapes, log_blowup, log_arity);
                            let proof = fri_prove(&fc, input, p_challenger, log_max_height);

                            (fc, proof, v_challenger)
                        },
                        |(fc, proof, mut v_challenger)| {
                            verifier::verify(
                                &TwoAdicFriGenericConfig::<Vec<(usize, Challenge)>, ()>(
                                    PhantomData,
                                ),
                                &fc,
                                &proof,
                                &mut v_challenger,
                                |_index, proof| Ok(proof.clone()),
                            )
                            .unwrap();
                        },
                    )
                },
            );
        }
    }

    for log_step_by in 0..3 {
        let mut test_shapes: [_; TEST_SHAPE_LEN] =
            array::from_fn(|step| test_shape_start - (step << log_step_by) as i32);
        test_shapes.reverse();
        for log_arity in [1usize, 2, 4] {
            let (fc, input, (p_challenger, _v_challenger), log_max_height) =
                fri_setup(test_shapes, log_blowup, log_arity);
            let proof = fri_prove(&fc, input, p_challenger, log_max_height);
            let proof_size = bincode::serialize(&proof).unwrap().len();
            println!(
                "fri/proof_size/test_shapes={:?}/log_arity={}:   proofsize:   [{:.2} KB {:.2} KB {:.2} KB]",
                test_shapes,
                log_arity,
                proof_size as f64 / 1024.0,
                proof_size as f64 / 1024.0,
                proof_size as f64 / 1024.0,
            );
        }
    }
}

criterion_group!(benches, bench_fri);
criterion_main!(benches);
