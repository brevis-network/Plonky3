use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize",
    deserialize = "Witness: Deserialize<'de>, InputProof: Deserialize<'de>"
))]
pub struct FriProof<F: Field, M: Mmcs<F>, Witness, InputProof> {
    /// soundcalc: One commitment per FRI commit-phase layer.
    /// soundcalc: Each element is the Merkle/PCS commitment of the extension-field codeword at that layer.
    pub commit_phase_commits: Vec<M::Commitment>,
    /// soundcalc: One entry per FRI query.
    /// soundcalc: Each `QueryProof` bundles:
    /// soundcalc:   (1) `input_proof`: openings in the base-field LDE MMCS, used to assemble the batched quotient r(X);
    /// soundcalc:   (2) `commit_phase_openings`: Merkle openings in every FRI layer at the queried index.
    pub query_proofs: Vec<QueryProof<F, M, InputProof>>,
    // This could become Vec<FC::Challenge> if this library was generalized to support non-constant
    // final polynomials.
    /// soundcalc: The value of the final constant polynomial after all FRI foldings.
    pub final_poly: F,
    /// soundcalc: Proof-of-work witness used to bound grinding attacks on the FRI challenges.
    pub pow_witness: Witness,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "InputProof: Serialize",
    deserialize = "InputProof: Deserialize<'de>",
))]
pub struct QueryProof<F: Field, M: Mmcs<F>, InputProof> {
    /// soundcalc: Base-field MMCS / LDE openings at this FRI query index.
    /// soundcalc: This ties the batched quotient r(X) back to the original LDE codewords
    pub input_proof: InputProof,
    /// For each commit phase commitment, this contains openings of a commit phase codeword at the
    /// queried location, along with an opening proof.
    /// soundcalc: The vector here is for each FRI commit-phase layer.
    pub commit_phase_openings: Vec<CommitPhaseProofStep<F, M>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    /// soundcalc: MMCS opening proof (e.g. Merkle path) witnessing that (value,`sibling_value`)
    /// soundcalc: is consistent with the commitment in `commit_phase_commits` for this layer.
    /// The opening of the commit phase codeword at the sibling location.
    // This may change to Vec<FC::Challenge> if the library is generalized to support other FRI
    // folding arities besides 2, meaning that there can be multiple siblings.
    pub sibling_value: F,

    pub opening_proof: M::Proof,
}
