use ndarray::{Array2, Zip};
use ndarray::prelude::*;
use ndarray::parallel::prelude::*;
use std::mem;
use ndarray_linalg::{Norm, c64};
use nalgebra::ComplexField;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub enum Error {
    DivBy0,
    BadIn,
    MultipleSolution,
    ComplexNumber,
    NotEnoughIterations,
}

#[derive(PartialEq, Eq)]
pub enum FactorizationType {
    Cholesky,
    Crout,
    Doolittle,
}

pub enum IterationType {
    Jacobi,
    GaussSeidel,
    SOR(f64),
}

#[derive(Debug, Serialize)]
pub struct LUStage<T: ComplexField> {
    l: Array2<T>,
    u: Array2<T>,
    k: usize,
    marks: Option<Array1<usize>>,
}

#[derive(Debug, Serialize)]
pub struct LUStages<T: ComplexField> {
    stages: Vec<LUStage<T>>,
}

impl <T: ComplexField> LUStages<T> {
    pub fn new() -> LUStages<f64> {
        LUStages {
            stages: Vec::<LUStage<f64>>::new(),
        }
    }

    pub fn new_with_complex() -> LUStages<c64> {
        LUStages {
            stages: Vec::<LUStage<c64>>::new(),
        }
    }

    pub fn registry(&mut self, l: &Array2<T>, u: &Array2<T>, k: usize) {
        self.stages.push(LUStage{
            l: l.clone(),
            u: u.clone(),
            k,
            marks: None,
        })
    }

    pub fn registry_with_marks(&mut self, l: &Array2<T>, u: &Array2<T>, k: usize, _marks: &Array1<usize>) {
        self.stages.push(LUStage{
            l: l.clone(),
            u: u.clone(),
            k,
            marks: Some(_marks.clone()),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct Stage {
    a: Array2<f64>,
    b: Array1<f64>,
    mults: Array1<f64>,
    k: usize,
    marks: Option<Array1<usize>>,
}

#[derive(Debug, Serialize)]
pub struct Stages {
    stages: Vec<Stage>,
}

impl Stages {
    pub fn new() -> Stages {
        Stages {
            stages: Vec::<Stage>::new(),
        }
    }

    pub fn registry(&mut self, a: &Array2<f64>, b: &Array1<f64>, mults: &Array1<f64>, k: usize) {
        self.stages.push(Stage {
            a: a.clone(),
            b: b.clone(),
            mults: mults.clone(),
            k,
            marks: None,
        })
    }

    pub fn registry_with_marks(&mut self, a: &Array2<f64>, b: &Array1<f64>,
                               mults: &Array1<f64>, k: usize, _marks: &Array1<usize>) {
        self.stages.push(Stage {
            a: a.clone(),
            b: b.clone(),
            mults: mults.clone(),
            k,
            marks: Some(_marks.clone()),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct Register {
    i: usize,
    err: f64,
    x: Array1<f64>,
}

#[derive(Debug, Serialize)]
pub struct Table {
    t: Array2<f64>,
    c: Array1<f64>,
    table: Vec<Register>,
}

impl Table {
    pub fn new() -> Table{
        Table {
            t: array![[0.]],
            c: array![0.],
            table: Vec::<Register>::new(),
        }
    }

    pub fn set_t_c(&mut self, t: &Array2<f64>, c: &Array1<f64>) {
        self.t = t.clone();
        self.c = c.clone();
    }

    pub fn registry(&mut self, i: usize, err: f64, x: &Array1<f64>) {
        self.table.push(Register {
            i,
            err,
            x: x.clone()
        });
    }
}

pub(crate) fn swap_rows(m: &mut Array2<f64>, a: usize, b: usize, start: usize) {
    let (mut ra, mut rb) = m.multi_slice_mut((s![a, start..], s![b, start..]));
    Zip::from(&mut ra).and(&mut rb).par_apply(|ea, eb| mem::swap(ea, eb));
}

pub(crate) fn swap_entire_cols(m: &mut Array2<f64>, a: usize, b: usize) {
    let (mut ca, mut cb) = m.multi_slice_mut((s![.., a], s![.., b]));
    Zip::from(&mut ca).and(&mut cb).par_apply(|ea, eb| mem::swap(ea, eb));
}

pub fn spectral_radius(m: &Array2<f64>) -> Result<f64, Error> {
    if !m.is_square() {
        return Err(Error::BadIn);
    }
    let n = m.nrows();
    let mut b_k = Array1::<f64>::ones(n);
    let mut b_k1 = m.dot(&b_k);
    b_k = &b_k1 / b_k1.norm();
    for _ in 0..10 {
        b_k1 = m.dot(&b_k);
        b_k = &b_k1 / b_k1.norm();

    }
    Ok(b_k.dot(&b_k1) / b_k.dot(&b_k))
}



