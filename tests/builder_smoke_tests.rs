extern crate fftw3;
extern crate num;
extern crate rand;

use rand::random;
use num::Float;
use num::Complex;
use fftw3::builder::{Planner, Direction};
use fftw3::builder::Rigor::{Estimate, Measure, Patient, Exhaustive};

fn almost_eq(x: &[f64], y: &[f64]) -> bool {
    x.len() == y.len() &&
        x.iter().zip(y.iter()).all(|(a,b)| (*a - *b).abs() <= 1e-5)
}
fn almost_eq_c(x: &[Complex<f64>], y: &[Complex<f64>]) -> bool {
    x.len() == y.len() &&
        x.iter().zip(y.iter()).all(|(a,b)| (*a - *b).norm() <= 1e-5)
}

const N: usize = 32;

macro_rules! smoke_test {
    ($forward: ident, $inverse: ident, $ctor: expr,
     $n: expr,
     $scale: expr,
     $cmp: expr) => {{
        for &rigor in [Estimate, Measure, Patient, Exhaustive].iter() {
            let in_ = fftw3::FftwVec::zeros(N);
            let out = fftw3::FftwVec::zeros(N * 10);
            let mut plan = Planner::new()
                .rigor(rigor)
                .$forward(in_, out)
                .plan().ok().unwrap();
            let in_ = fftw3::FftwVec::zeros($n);
            let out = fftw3::FftwVec::zeros(N * 10);
            let mut inv = Planner::new()
                .direction(Direction::Backward)
                .rigor(rigor)
                .$inverse(in_, out)
                .plan().ok().unwrap();

            for x in plan.input().iter_mut() {
                *x = $ctor
            }

            plan.execute();

            {
                let invin = inv.input();
                let pout = plan.output().unwrap();
                for i in 0 .. invin.len() {
                    invin[i] = pout[i];
                }
            }
            inv.execute();
            let scale = 1.0 / plan.input().len() as f64;
            for x in inv.output().unwrap().iter_mut() {
                *x = $scale(x, scale)
            }
            assert!($cmp(plan.input(), &inv.output().unwrap()[..N]))
        }
    }}
}

#[test]
fn c2c_smoke_test() {
    smoke_test!(c2c, c2c, Complex::new(random(), random()), N,
                |x: &Complex<_>, scale| x.scale(scale), almost_eq_c)
}
#[test]
fn r2c_c2r_smoke_test() {
    smoke_test!(r2c, c2r, random(), N / 2 + 1, |x: &f64, scale| *x * scale, almost_eq)
}

#[test]
fn c2c_inplace_smoke_test() {
    for &rigor in [Estimate, Measure, Patient, Exhaustive].iter() {
        let data = (0..N).map(|_| Complex::new(random(), random())).collect::<Vec<_>>();
        let a = fftw3::FftwVec::zeros(N);
        let b = fftw3::FftwVec::zeros(N);
        let mut plan = Planner::new()
            .rigor(rigor)
            .inplace()
            .c2c(a)
            .plan().ok().unwrap();
        {
            let pin = plan.input();
            for i in 0 .. pin.len() {
                pin[i] = data[i];
            }
        }
        plan.execute();

        let mut inv = Planner::new()
            .rigor(rigor)
            .direction(Direction::Backward)
            .inplace()
            .c2c(b)
            .plan().ok().unwrap();

        {
            let invin = inv.input();
            let pout = plan.input();
            for i in 0 .. invin.len() {
                invin[i] = pout[i];
            }
        }

        inv.execute();
        for x in inv.input().iter_mut() {
            *x = x.unscale(N as f64);
        }
        assert!(almost_eq_c(inv.input(), &*data));
    }
}
