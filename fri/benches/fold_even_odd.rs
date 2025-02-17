use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::extension::Complex;
use p3_field::TwoAdicField;
use p3_fri::fold_even_odd;
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench<F: TwoAdicField>(c: &mut Criterion, log_sizes: &[usize], field_name: &'static str)
where
    Standard: Distribution<F>,
{
    let name = format!("fold_even_odd::<{}>", field_name,);
    let mut group = c.benchmark_group(&name);
    group.sample_size(10);

    for log_size in log_sizes {
        let n = 1 << log_size;

        group.bench_function(format!("({})", log_size), |b| {
            b.iter_with_setup(
                || {
                    let mut rng = thread_rng();
                    let beta = rng.sample(Standard);
                    let poly = rng.sample_iter(Standard).take(n).collect_vec();
                    let poly = RowMajorMatrix::new(poly, 2);
                    (poly, beta)
                },
                |(poly, beta)| {
                    fold_even_odd(poly, beta);
                },
            )
        });
    }
}

fn bench_fold_even_odd(c: &mut Criterion) {
    let log_sizes = [20, 24];

    bench::<BabyBear>(c, &log_sizes, "BabyBear");
    bench::<Goldilocks>(c, &log_sizes, "Goldilocks");
    bench::<Complex<Mersenne31>>(c, &log_sizes, "Complex<Mersenne31>");
}

criterion_group!(benches, bench_fold_even_odd);
criterion_main!(benches);
