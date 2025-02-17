use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{
    CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof, NormalizeQueryProof, QueryProof,
};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize) -> G::InputProof,
) -> FriProof<Challenge, M, Challenger::Witness, G::InputProof>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    // check sorted descending
    assert!(inputs
        .iter()
        .tuple_windows()
        .all(|(l, r)| l.len() >= r.len()));

    let log_max_height = log2_strict_usize(inputs[0].len());

    let normalize_phase_result = normalize_phase(g, config, &inputs, challenger);
    let commit_phase_result = commit_phase(
        g,
        config,
        normalize_phase_result.normalized_inputs,
        challenger,
    );

    let pow_witness = challenger.grind(config.proof_of_work_bits);
    let query_indices =
        iter::repeat_with(|| challenger.sample_bits(log_max_height + g.extra_query_index_bits()))
            .take(config.num_queries)
            .collect_vec();

    let normalize_query_proofs = info_span!("normalize query phase").in_scope(|| {
        query_indices
            .iter()
            .map(|&index| NormalizeQueryProof {
                normalize_phase_openings: normalize_phase_result
                    .data
                    .iter()
                    .zip_eq(
                        normalize_phase_result
                            .commits
                            .iter()
                            .map(|(_, height)| *height),
                    )
                    .map(|(data, height)| {
                        let shift = (height - config.log_blowup) % config.log_arity;
                        answer_query_single_step(
                            config,
                            data,
                            index >> g.extra_query_index_bits() >> (log_max_height - height),
                            shift,
                        )
                    })
                    .collect(),
            })
            .collect()
    });

    let query_proofs = info_span!("query phase").in_scope(|| {
        let shift = (log_max_height - config.log_blowup) % config.log_arity;

        query_indices
            .into_iter()
            .map(|index| QueryProof {
                input_proof: open_input(index),
                commit_phase_openings: answer_query(
                    config,
                    &commit_phase_result.data,
                    index >> g.extra_query_index_bits() >> shift,
                ),
            })
            .collect()
    });

    FriProof {
        commit_phase_commits: commit_phase_result.commits,
        normalize_phase_commits: normalize_phase_result.commits,
        normalize_query_proofs,
        query_proofs,
        final_poly: commit_phase_result.final_poly,
        pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_poly: F,
}

#[instrument(name = "normalize phase", skip_all)]
fn normalize_phase<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: &[Vec<Challenge>],
    challenger: &mut Challenger,
) -> NormalizePhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: TwoAdicField + ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: CanObserve<M::Commitment> + FieldChallenger<Val>,
    G: FriGenericConfig<Challenge>,
{
    let mut commits = vec![];
    let mut data = vec![];

    let mut normalized_inputs: [Option<Vec<Challenge>>; 32] = [const { None }; 32];
    inputs.iter().for_each(|input| {
        let log_height = log2_strict_usize(input.len());
        let num_folds = (log_height - config.log_blowup) % config.log_arity;
        let input = if log_height >= config.log_blowup
            && (log_height - config.log_blowup) % config.log_arity == 0
        {
            input.clone()
        } else {
            let current = input.clone();
            let leaves = RowMajorMatrix::new(current, 1 << num_folds);

            let (commit, prover_data) = config.mmcs.commit_matrix(leaves.clone());
            challenger.observe(commit.clone());
            commits.push((commit, log_height));
            data.push(prover_data);

            let beta: Challenge = challenger.sample_ext_element();
            g.fold_matrix(beta, leaves.as_view(), num_folds)
        };
        match &mut normalized_inputs[log_height - num_folds] {
            Some(v) => {
                v.iter_mut()
                    .zip_eq(input)
                    .for_each(|(v_elem, c_elem)| *v_elem += c_elem);
            }
            None => {
                normalized_inputs[log_height - num_folds] = Some(input);
            }
        };
    });

    let normalized_inputs = normalized_inputs.into_iter().rev().flatten().collect();

    NormalizePhaseResult {
        commits,
        data,
        normalized_inputs,
    }
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    // By the time the prover gets to this phase, the `Some` inputs must all come from polynomials
    // whose log-degree is a multiple of `config.log_arity`.
    debug_assert!(inputs
        .iter()
        .all(|x| (log2_strict_usize(x.len()) - config.log_blowup) % config.log_arity == 0));

    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];

    while folded.len() > config.blowup() {
        // A row of `leaves` is the information necessary to open the folded polynomial at a given
        // index.
        let leaves = RowMajorMatrix::new(folded.clone(), 1 << config.log_arity);
        let (commit, prover_data) = config.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());

        let beta: Challenge = challenger.sample_ext_element();
        // We passed ownership of `current` to the MMCS, so get a reference to it
        let leaves = config.mmcs.get_matrices(&prover_data).pop().unwrap();
        folded = g.fold_matrix(beta, leaves.as_view(), config.log_arity);

        commits.push(commit);
        data.push(prover_data);

        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            izip!(&mut folded, v).for_each(|(c, x)| *c += x);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(folded.len(), config.blowup());
    let final_poly = folded[0];
    for x in folded {
        assert_eq!(x, final_poly);
    }
    challenger.observe_ext_element(final_poly);

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

/// A function to answer a single step of the query phases.
fn answer_query_single_step<F: Field, M: Mmcs<F>>(
    config: &FriConfig<M>,
    data: &M::ProverData<RowMajorMatrix<F>>,
    index: usize,
    log_num_leaves: usize,
) -> CommitPhaseProofStep<F, M> {
    let (mut opened_rows, opening_proof) = config.mmcs.open_batch(index >> log_num_leaves, data);

    assert_eq!(opened_rows.len(), 1);
    let opened_row = opened_rows.pop().unwrap();
    assert_eq!(
        opened_row.len(),
        1 << log_num_leaves,
        "Committed data should be in tuples of size arity."
    );

    let mut siblings = opened_row;
    siblings.remove(index & ((1 << log_num_leaves) - 1));

    CommitPhaseProofStep {
        sibling_values: siblings,
        opening_proof,
    }
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> Vec<CommitPhaseProofStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    let log_arity = config.log_arity;
    commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, data)| {
            let index_i = index >> (i * log_arity);
            answer_query_single_step(config, data, index_i, log_arity)
        })
        .collect()
}

struct NormalizePhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<(M::Commitment, usize)>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    normalized_inputs: Vec<Vec<F>>,
}
