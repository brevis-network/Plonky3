use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr, InputError> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    OpenedRowMismatch,
    InputError(InputError),
    FinalPolyMismatch,
    InvalidPowWitness,
}

pub fn verify<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    proof: &FriProof<Challenge, M, Challenger::Witness, G::InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize, &G::InputProof) -> Result<Vec<(usize, Challenge)>, G::InputError>,
    log_max_height: usize,
) -> Result<(), FriError<M::Error, G::InputError>>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();

    // Observe all coefficients of the final polynomial.
    proof
        .final_poly
        .iter()
        .for_each(|x| challenger.observe_ext_element(*x));

    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    for qp in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height + g.extra_query_index_bits());
        let ro = open_input(index, &qp.input_proof).map_err(FriError::InputError)?;

        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        let folded_eval = verify_query(
            g,
            config,
            index >> g.extra_query_index_bits(),
            izip!(
                betas.clone(),
                &proof.commit_phase_commits,
                &qp.commit_phase_openings
            ),
            ro,
            log_max_height,
        )?;

        // Even though we might do a partial fold at the end, it's okay to shift by arity_bits since in that case
        // final_poly_index is zero
        let final_poly_index = index >> (proof.commit_phase_commits.len() * config.arity_bits);

        let mut eval = Challenge::ZERO;

        // We open the final polynomial at index `final_poly_index`, which corresponds to evaluating
        // the polynomial at x^k, where x is the 2-adic generator of order `max_height` and k is
        // `reverse_bits_len(final_poly_index, log_max_height)`.
        let x = Challenge::two_adic_generator(log_max_height)
            .exp_u64(reverse_bits_len(final_poly_index, log_max_height) as u64);
        let mut x_pow = Challenge::ONE;

        // Evaluate the final polynomial at x.
        for coeff in &proof.final_poly {
            eval += *coeff * x_pow;
            x_pow *= x;
        }

        if eval != folded_eval {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

type CommitStep<'a, F, M> = (
    F,
    &'a <M as Mmcs<F>>::Commitment,
    &'a CommitPhaseProofStep<F, M>,
);

fn verify_query<'a, G, F, M>(
    g: &G,
    config: &FriConfig<M>,
    mut index: usize,
    mut steps: impl Iterator<Item = CommitStep<'a, F, M>>,
    reduced_openings: Vec<(usize, F)>,
    log_max_height: usize,
) -> Result<F, FriError<M::Error, G::InputError>>
where
    F: Field + TwoAdicField,
    M: Mmcs<F> + 'a,
    G: FriGenericConfig<F>,
{
    let mut folded_eval = F::ZERO;
    let roots_of_unity = (1..=config.arity_bits)
        .map(|i| F::two_adic_generator(i))
        .collect::<Vec<_>>();
    let mut x = F::two_adic_generator(log_max_height)
        .exp_u64(reverse_bits_len(index, log_max_height) as u64);

    let mut ro_iter = reduced_openings.into_iter().peekable();
    let mut log_folded_height = log_max_height;

    while log_folded_height > config.log_blowup + config.log_final_poly_len {
        let cur_arity_bits = config.arity_bits.min(log_folded_height);

        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            folded_eval += ro;
        }

        let (mut beta, comm, opening) = steps.next().unwrap();

        let index_row = index >> cur_arity_bits;

        // Verify that `folded_eval` and evals from reduced_openings match opened_rows
        if folded_eval != opening.opened_rows[0][index % (1 << cur_arity_bits)] {
            return Err(FriError::OpenedRowMismatch);
        }
        for row in opening.opened_rows.iter().skip(1) {
            let (lh, ro) = ro_iter.next().unwrap();
            // Make sure the read row is of the correct length
            if row.len() != 1 << (lh + cur_arity_bits - log_folded_height) {
                return Err(FriError::OpenedRowMismatch);
            }

            let current_index = index >> (log_folded_height - lh);
            if ro != row[current_index % row.len()] {
                return Err(FriError::OpenedRowMismatch);
            }
        }

        let dims: Vec<_> = opening
            .opened_rows
            .iter()
            .map(|opened_row| Dimensions {
                width: opened_row.len(),
                height: 1 << (log_folded_height - cur_arity_bits),
            })
            .collect();

        config
            .mmcs
            .verify_batch(
                comm,
                &dims,
                index_row,
                &opening.opened_rows,
                &opening.opening_proof,
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        // Do the folding logic

        let mut opened_rows_iter = opening.opened_rows.iter().peekable();
        let mut folded_row = opened_rows_iter.next().unwrap().clone();
        reverse_slice_index_bits(&mut folded_row);

        let mut x_prime = x * roots_of_unity[cur_arity_bits - 1]
            .inverse()
            .exp_u64(reverse_bits_len(index & ((1 << cur_arity_bits) - 1), cur_arity_bits) as u64);

        while folded_row.len() > 1 {
            index >>= 1;
            log_folded_height -= 1;

            folded_row = fold_partial_row(g, x_prime, &roots_of_unity, beta, folded_row);
            beta = beta.square();
            x = x.square();
            x_prime = x_prime.square();

            if let Some(poly_eval) = opened_rows_iter.next_if(|v| v.len() == folded_row.len()) {
                let mut poly_eval = poly_eval.to_vec();
                reverse_slice_index_bits(&mut poly_eval);
                izip!(&mut folded_row, poly_eval).for_each(|(f, v)| *f += v);
            }
        }

        folded_eval = folded_row.pop().unwrap();
        assert!(folded_row.is_empty());
    }

    debug_assert!(
        index < config.blowup() * config.final_poly_len(),
        "index was {}",
        index,
    );
    debug_assert!(
        ro_iter.next().is_none(),
        "verifier reduced_openings were not in descending order?"
    );

    Ok(folded_eval)
}

fn fold_partial_row<G, F>(g: &G, mut x: F, roots_of_unity: &[F], beta: F, evals: Vec<F>) -> Vec<F>
where
    G: FriGenericConfig<F>,
    F: Field,
{
    let n_bits = log2_strict_usize(evals.len()) - 1;
    let half_len = evals.len() >> 1;
    izip!(&evals, &evals[half_len..])
        .map(|(&eval_0, &eval_1)| {
            let res = g.fold_row_with_x(roots_of_unity[0], x, beta, [eval_0, eval_1].into_iter());
            x *= roots_of_unity[n_bits];
            res
        })
        .collect()
}
