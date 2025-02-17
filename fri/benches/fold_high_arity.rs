use std::marker::PhantomData;

use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::extension::Complex;
use p3_field::TwoAdicField;
use p3_fri::{FriGenericConfig, TwoAdicFriGenericConfig};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench<F: TwoAdicField>(
    c: &mut Criterion,
    log_sizes: &[usize],
    log_arities: &[usize],
    field_name: &'static str,
) where
    Standard: Distribution<F>,
{
    let name = format!("fold::<{}>", field_name,);
    let mut group = c.benchmark_group(&name);
    group.sample_size(10);
    let g: TwoAdicFriGenericConfig<(), ()> = TwoAdicFriGenericConfig(PhantomData);

    for log_size in log_sizes {
        for log_arity in log_arities {
            let n = 1 << log_size;

            group.bench_function(format!("({}, {})", log_size, log_arity), |b| {
                b.iter_with_setup(
                    || {
                        let mut rng = thread_rng();
                        let beta = rng.sample(Standard);
                        let poly = rng.sample_iter(Standard).take(n).collect_vec();
                        let poly = RowMajorMatrix::new(poly, 1 << log_arity);
                        (poly, beta)
                    },
                    |(poly, beta)| {
                        g.fold_matrix(beta, poly, *log_arity);
                    },
                )
            });
        }
    }
}

fn bench_fold_high_arity(c: &mut Criterion) {
    let log_sizes = [20, 24];
    let log_arities = [1, 2, 4];

    bench::<BabyBear>(c, &log_sizes, &log_arities, "BabyBear");
    bench::<Goldilocks>(c, &log_sizes, &log_arities, "Goldilocks");
    bench::<Complex<Mersenne31>>(c, &log_sizes, &log_arities, "Complex_Mersenne31");
}

criterion_group!(benches, bench_fold_high_arity);
criterion_main!(benches);
