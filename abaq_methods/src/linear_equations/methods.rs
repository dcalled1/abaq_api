use num_traits::{Zero, FromPrimitive};
use nalgebra::{ComplexField, Complex};
use crate::linear_equations::utilities::{Error, swap_rows, swap_entire_cols, FactorizationType, IterationType, spectral_radius, Stages, LUStages, Table};
use ndarray::{Array2, Array1, Zip};
use ndarray::prelude::*;
use ndarray::parallel::prelude::*;
use num_traits::float::FloatCore;
use ndarray_linalg::norm::*;
use ndarray_linalg::{Inverse, c64, Determinant};

fn back_substitution<T: ComplexField>(a: &Array2<T>, b: &Array1<T>) -> Result<Array1<T>, Error> {
    if !a.is_square() {
        return Err(Error::BadIn);
    }
    let n = b.len();
    let mut x = Array1::<T>::zeros(n);
    if a[(n - 1, n - 1)].is_zero() {
        return Err(Error::DivBy0);
    }
    x[n-1] = b[n-1]/ a[(n - 1, n - 1)];
    let mut sum: T;
    for i in (0..=n-2).rev() {
        sum = T::zero();
        for p in (i + 1) ..= (n - 1) {
            sum += a[(i, p)] * x[p];
        }
        if a[(i, i)].is_zero() {
            return Err(Error::DivBy0);
        }
        x[i] = (b[i] - sum)/ a[(i, i)];

    }
    println!("{}", x);

    Ok(x)

}

pub(crate) fn forward_substitution<T: ComplexField>(a: &Array2<T>, b: &Array1<T>) -> Result<Array1<T>, Error> {
    if !a.is_square() {
        return Err(Error::BadIn);
    }
    let n = b.len();
    let mut x = Array1::<T>::zeros(n);
    if a[(0, 0)].is_zero() {
        return Err(Error::DivBy0);
    }
    x[0] = b[0]/ a[(0, 0)];
    let mut sum: T;
    for i in 1..=n-1 {
        sum = T::zero();
        for p in 0..= (i - 1) {
            sum += a[(i, p)] * x[p];
        }
        if a[(i, i)].is_zero() {
            return Err(Error::DivBy0);
        }
        x[i] = (b[i] - sum)/ a[(i, i)];

    }
    println!("{}", x);

    Ok(x)
}

fn eliminate(a: &mut Array2<f64>, b: &mut Array1<f64>, k: usize) -> Array1<f64>{
    let row_k = Zip::from(a.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
    let bk = b[k];
    Zip::from(a.slice_mut(s![k+1.., k..]).genrows_mut()).and(b.slice_mut(s![k+1..]))
        .par_apply_collect(|mut row_i, mut bi| {
            let mul = row_i[0]/row_k[0];
            Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
            *bi -= mul * bk;
            mul
        })
}

pub fn simple_elimination(a: &Array2<f64>, b: &Array1<f64>) -> (Result<(Array2<f64>, Array1<f64>), Error>, Stages) {
    let mut stages = Stages::new();
    if !a.is_square() {
        return (Err(Error::BadIn), stages);
    }
    let n = a.nrows();
    if n != b.len() {
        return (Err(Error::BadIn), stages);
    }
    if a.det().unwrap() == 0. {
        return (Err(Error::MultipleSolution), stages);
    }
    let mut new_a = a.clone();
    let mut new_b = b.clone();
    println!("{}", new_a);
    for k in 0..n-1 {
        if new_a[[k, k]] == 0. {
            for i in k+1..n {
                if new_a[[i, k]] != 0. {
                    swap_rows(&mut new_a, i, k, k);
                    break;
                }
            }
        }
        let mults = eliminate(&mut new_a, &mut new_b, k);
        stages.registry(&new_a, &new_b, &mults, k);
        //println!("k: {}\nU: \n{}\nmults:\n{}\n----------", k, new_a, mults);
    }
    (Ok((new_a, new_b)), stages)
}

pub fn gaussian_elimination(a: &Array2<f64>, b: &Array1<f64>) -> (Result<Array1<f64>, Error>, Stages) {
    let (u, b_, stages) = match simple_elimination(&a, &b) {
        (Ok((_u, _b_)), _stages) => (_u, _b_, _stages),
        (Err(e), _stages) => return (Err(e), _stages),
    };
    let x = match back_substitution(&u, &b_) {
        Ok(x_) => x_,
        Err(e) => return (Err(e), stages),
    };
    (Ok(x), stages)
}

pub fn elimination_with_partial_pivoting(a: &Array2<f64>, b: &Array1<f64>) -> (Result<(Array2<f64>, Array1<f64>), Error>, Stages) {
    let mut stages = Stages::new();
    if !a.is_square() {
        return (Err(Error::BadIn), stages);
    }
    let n = a.nrows();
    if n != b.len() {
        return (Err(Error::BadIn), stages);
    }
    if a.det().unwrap() == 0. {
        return (Err(Error::MultipleSolution), stages);
    }
    let mut new_a = a.clone();
    let mut new_b = b.clone();
    println!("{}", new_a);
    for k in 0..n-1 {
        let (mut max, mut max_row) = (new_a[(k, k)].abs(), k);
        for s in k+1..n {
            let tmp = new_a[(s, k)].abs();
            if tmp > max {
                max = tmp;
                max_row = s;
            }
        }
        if max == 0. {
            return (Err(Error::MultipleSolution), stages);
        }
        if max_row != k {
            swap_rows(&mut new_a, max_row, k, k);
            new_b.swap(max_row, k);
        }
        let mults = eliminate(&mut new_a, &mut new_b, k);
        stages.registry(&new_a, &new_b, &mults, k);
        println!("k: {}\nU: \n{}\nmults:\n{}\n----------", k, new_a, mults);
    }
    (Ok((new_a, new_b)), stages)
}

pub fn gaussian_elimination_partial_pivoting(a: &Array2<f64>, b: &Array1<f64>) -> (Result<Array1<f64>, Error>, Stages) {
    let (u, b_, stages) = match elimination_with_partial_pivoting(&a, &b) {
        (Ok((_u, _b_)), _stages) => (_u, _b_, _stages),
        (Err(e), _stages) => return (Err(e), _stages),
    };
    let x = match back_substitution(&u, &b_) {
        Ok(x_) => x_,
        Err(e) => return (Err(e), stages),
    };
    (Ok(x), stages)
}

pub fn elimination_with_total_pivoting(a: &Array2<f64>, b: &Array1<f64>) -> (Result<(Array2<f64>, Array1<f64>, Array1<usize>), Error>, Stages) {
    let mut stages = Stages::new();
    if !a.is_square() {
        return (Err(Error::BadIn), stages);
    }
    let n = a.nrows();
    if n != b.len() {
        return (Err(Error::BadIn), stages);
    }
    if a.det().unwrap() == 0. {
        return (Err(Error::MultipleSolution), stages);
    }
    let mut new_a = a.clone();
    let mut new_b = b.clone();
    let mut marks = Array1::<usize>::from((0..n).collect::<Vec<usize>>());
    println!("{} \nmarks:\n{}", new_a, marks);
    for k in 0..n-1 {
        let (mut max, mut max_row, mut max_col) = (0., k, k);
        for r in k..n {
            for s in k..n {
                let tmp = new_a[(r, s)].abs();
                if tmp > max {
                    max = tmp;
                    max_row = r;
                    max_col = s;
                }
            }
        }
        if max == 0. {
            return (Err(Error::MultipleSolution), stages);
        }
        if max_row != k {
            swap_rows(&mut new_a, max_row, k, k);
            new_b.swap(max_row, k);
        }
        if max_col != k {
            swap_entire_cols(&mut new_a, max_col, k);
            marks.swap(max_col, k);
        }
        let mults = eliminate(&mut new_a, &mut new_b, k);
        stages.registry_with_marks(&new_a, &new_b, &mults, k, &marks);
        println!("k: {}\nU: \n{}\nmults:\n{}\nmarks:\n{}\n----------", k, new_a, mults, marks);
    }
    (Ok((new_a, new_b, marks)), stages)
}

pub fn short_by_marks(v: &Array1<f64>, marks: &Array1<usize>) -> Array1<f64> {
    Zip::from(marks).par_apply_collect(|mark| v[*mark])
}

pub fn gaussian_elimination_total_pivoting(a: &Array2<f64>, b: &Array1<f64>)
                -> (Result<Array1<f64>, Error>, Stages) {
    let (u, new_b, marks, stages) =
        match elimination_with_total_pivoting(&a, &b) {
        (Ok((_u, _b_, _marks)), _stages) => (_u, _b_, _marks, _stages),
        (Err(e), _stages) => return (Err(e), _stages),
    };
    let b_ = short_by_marks(&new_b, &marks);
    let x = match back_substitution(&u, &b_) {
        Ok(x_) => x_,
        Err(e) => return (Err(e), stages),
    };
    (Ok(x), stages)

}

pub fn simple_elimination_lu(m: &Array2<f64>) -> (Result<(Array2<f64>, Array2<f64>), Error>, LUStages<f64>) {
    let mut stages = LUStages::<f64>::new();
    let n = m.nrows();
    let mut mults = Array2::<f64>::eye(n);
    let mut new_m = Array2::<f64>::zeros((n, n));
    new_m.clone_from(m);
    println!("{}", new_m);
    for k in 0..n-1 {
        let row_k = Zip::from(new_m.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
        Zip::from(new_m.slice_mut(s![k+1.., k..]).genrows_mut())
            .and(mults.slice_mut(s![k+1.., k]))
            .par_apply(|mut row_i, mut mul_slot| {
            let mul = row_i[0]/row_k[0];
            *mul_slot = mul;
            Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
        });

        stages.registry(&mults, &new_m, k);
        /*let mut mul: f64;
        for i in k+1..n {
            mul = new_m[(i, k)]/ new_m[(k, k)];
            mults[(i, k)] = mul;
            for j in k..n {
                new_m[(i, j)] -= mul * new_m[(k, j)]
            }
        }*/
        println!("k: {}\nU: \n{}\nmults:\n{}\n----------", k, new_m, mults);
    }
    (Ok((new_m, mults)), stages)
}

pub fn gaussian_factorization(a: &Array2<f64>, b: &Array1<f64>) -> (Result<Array1<f64>, Error>, LUStages<f64>) {
    if !a.is_square() {
        return (Err(Error::BadIn), LUStages::<f64>::new());
    }
    if a.nrows() != b.len() {
        return (Err(Error::BadIn), LUStages::<f64>::new());
    }
    if a.det().unwrap() == 0. {
        return (Err(Error::MultipleSolution), LUStages::<f64>::new());
    }
    let (l, u, stages) = match simple_elimination_lu(a) {
        (Ok((_l, _u)), _stages) => (_l, _u, _stages),
        (Err(e), _stages) => return (Err(e), _stages),
    };
    let z = match forward_substitution(&l, &b) {
        Ok(_z) => _z,
        Err(e) => return (Err(e), stages),
    };
    let x = match back_substitution(&u, &z) {
        Ok(_x) => _x,
        Err(e) => return (Err(e), stages),
    };
    (Ok(x), stages)
}

pub fn pivoting_elimination_lu(m: &Array2<f64>) -> (Result<(Array2<f64>, Array2<f64>, Array1<usize>), Error>, LUStages<f64>) {
    let mut stages = LUStages::<f64>::new();
    let n = m.nrows();
    let mut mults = Array2::<f64>::eye(n);
    let mut new_m = Array2::<f64>::zeros((n, n));
    new_m.clone_from(m);
    let mut marks = Array1::<usize>::from((0..n).collect::<Vec<usize>>());
    println!("{}", new_m);
    for k in 0..n-1 {
        let (mut max, mut max_row) = (new_m[(k, k)].abs(), k);
        for s in k+1..n {
            let tmp = new_m[(s, k)].abs();
            if tmp > max {
                max = tmp;
                max_row = s;
            }
        }
        if max == 0. {
            return (Err(Error::MultipleSolution), stages);
        }
        if max_row != k {
            swap_rows(&mut new_m, max_row, k, k);
            marks.swap(max_row, k);
        }
        let row_k = Zip::from(new_m.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
        Zip::from(new_m.slice_mut(s![k+1.., k..]).genrows_mut())
            .and(mults.slice_mut(s![k+1.., k]))
            .par_apply(|mut row_i, mut mul_slot| {
                let mul = row_i[0]/row_k[0];
                *mul_slot = mul;
                Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
            });
        stages.registry_with_marks(&mults, &new_m, k,&marks);
        println!("k: {}\nU: \n{}\nmults:\n{}\nmarks:\n{}\n----------", k, new_m, mults, marks);
    }
    (Ok((new_m, mults, marks)), stages)
}

pub fn pivoting_gaussian_factorization(a: &Array2<f64>, b: &Array1<f64>) -> (Result<Array1<f64>, Error>, LUStages<f64>) {
    if !a.is_square() {
        return (Err(Error::BadIn), LUStages::<f64>::new());
    }
    if a.nrows() != b.len() {
        return (Err(Error::BadIn), LUStages::<f64>::new());
    }
    if a.det().unwrap() == 0. {
        return (Err(Error::MultipleSolution), LUStages::<f64>::new());
    }
    let (l, u, marks, stages) = match pivoting_elimination_lu(a){
        (Ok((_l, _u, _marks)), _stages) => (_l, _u, _marks, _stages),
        (Err(e), _stages) => return (Err(e), _stages),
    };
    let b_ = short_by_marks(b, &marks);
    let z = match forward_substitution(&l, &b_) {
        Ok(_z) => _z,
        Err(e) => return (Err(e), stages),
    };
    let x = match back_substitution(&u, &z) {
        Ok(_x) => _x,
        Err(e) => return (Err(e), stages),
    };
    (Ok(x), stages)
}

pub fn direct_factorization(m: &Array2<f64>, method: FactorizationType)
                        -> (Result<(Array2<f64>, Array2<f64>), Error>, LUStages<f64>) {
    let mut stages = LUStages::<f64>::new();
    if !m.is_square() {
        return (Err(Error::BadIn), stages);
    }
    let n = m.nrows();
    let (mut l, mut u)
                = (Array2::<f64>::eye(n), Array2::<f64>::eye(n));
    let mut sum: f64;
    for k in 0..n {
        sum = Zip::from(l.slice(s![k, ..k]))
            .and(u.slice(s![..k, k])).fold(0., |ac, el, eu| ac + el * eu );
        let val = m[(k, k)] - sum;
        if val.is_zero() {
            return (Err(Error::DivBy0), stages);
        }
        l[(k, k)] = match method {
            FactorizationType::Crout => val,
            FactorizationType::Doolittle => 1.,
            FactorizationType::Cholesky => {
                if val < 0. {
                    return (Err(Error::ComplexNumber), stages);
                }
                val.sqrt()
            },
        };
        u[(k, k)] = match method {
            FactorizationType::Doolittle => val,
            FactorizationType::Crout => 1.,
            FactorizationType::Cholesky => {
                if val < 0. {
                    return (Err(Error::ComplexNumber), stages);
                }
                val.sqrt()
            },
        };

        for i in k+1..n {
            sum = Zip::from(l.slice(s![i, ..k])).and(u.slice(s![..k, k]))
                .fold(0., |ac, el, eu| ac + el * eu );
            l[(i, k)] = (m[(i, k)]-sum)/u[(k, k)];
        }

        for i in k+1..n {
            sum = Zip::from(l.slice(s![k, ..k])).and(u.slice(s![..k, i]))
                .fold(0., |ac, el, eu| ac + el * eu );
            u[(k, i)] = (m[(k, i)]-sum)/l[(k, k)];
        }
        stages.registry(&l, &u, k);
        println!("-------------------\nk: {}\nL:\n{}\nU:{}", k, l, u);
    }

    (Ok((l, u)), stages)
}

pub fn direct_factorization_with_complex(m: &Array2<f64>)
    -> (Result<(Array2<Complex<f64>>, Array2<Complex<f64>>), Error>, LUStages<c64>) {
    let mut stages = LUStages::<c64>::new_with_complex();
    let n = m.nrows();
    let (mut l, mut u)
                = (Array2::<Complex<f64>>::eye(n), Array2::<Complex<f64>>::eye(n));
    let mut sum: Complex<f64>;
    for k in 0..n {
        sum = Zip::from(l.slice(s![k, ..k])).and(u.slice(s![..k, k]))
            .fold(Complex::<f64>::zero(), |ac, el, eu| ac + el * eu );
        let val = (Complex::<f64>::from_real(m[(k, k)]) - sum).sqrt();
        if val.is_zero() {
            return (Err(Error::DivBy0), stages);
        }
        l[(k, k)] = val;
        u[(k, k)] = val;

        for i in k+1..n {
            sum = Zip::from(l.slice(s![i, ..k])).and(u.slice(s![..k, k]))
                .fold(Complex::<f64>::zero(), |ac, el, eu| ac + el * eu );
            l[(i, k)] = (Complex::<f64>::from_real(m[(i, k)])-sum)/u[(k, k)];
        }

        for i in k+1..n {
            sum = Zip::from(l.slice(s![k, ..k])).and(u.slice(s![..k, i]))
                .fold(Complex::<f64>::zero(), |ac, el, eu| ac + el * eu );
            u[(k, i)] = (Complex::<f64>::from_real(m[(k, i)])-sum)/l[(k, k)];
        }
        stages.registry(&l, &u, k);
        println!("-------------------\nk: {}\nL:\n{}\nU:{}", k, l, u);
    }
    (Ok((l, u)), stages)
}

pub fn doolittle(a: &Array2<f64>, b: &Array1<f64>) -> (Result<Array1<f64>, Error>, LUStages<f64>) {
    if !a.is_square() {
        return (Err(Error::BadIn), LUStages::<f64>::new());
    }
    if a.nrows() != b.len() {
        return (Err(Error::BadIn), LUStages::<f64>::new());
    }
    if a.det().unwrap() == 0. {
        return (Err(Error::MultipleSolution), LUStages::<f64>::new());
    }
    let (l, u, stages) =
        match direct_factorization(&a, FactorizationType::Doolittle) {
        (Ok((_l, _u)), _stages) => (_l, _u, _stages),
        (Err(e), _stages) => return (Err(e), _stages),
    };
    let z = match forward_substitution(&l, &b) {
        Ok(_z) => _z,
        Err(e) => return (Err(e), stages),
    };
    let x = match back_substitution(&u, &z) {
        Ok(_x) => _x,
        Err(e) => return (Err(e), stages),
    };
    (Ok(x), stages)
}

pub fn crout(a: &Array2<f64>, b: &Array1<f64>) -> (Result<Array1<f64>, Error>, LUStages<f64>) {
    if !a.is_square() {
        return (Err(Error::BadIn), LUStages::<f64>::new());
    }
    if a.nrows() != b.len() {
        return (Err(Error::BadIn), LUStages::<f64>::new());
    }
    if a.det().unwrap() == 0. {
        return (Err(Error::MultipleSolution), LUStages::<f64>::new());
    }
    let (l, u, stages) =
        match direct_factorization(&a, FactorizationType::Crout) {
            (Ok((_l, _u)), _stages) => (_l, _u, _stages),
            (Err(e), _stages) => return (Err(e), _stages),
        };
    let z = match forward_substitution(&l, &b) {
        Ok(_z) => _z,
        Err(e) => return (Err(e), stages),
    };
    let x = match back_substitution(&u, &z) {
        Ok(_x) => _x,
        Err(e) => return (Err(e), stages),
    };
    (Ok(x), stages)
}

pub fn cholesky(a: &Array2<f64>, _b: &Array1<f64>) -> (Result<Array1<f64>, Error>, LUStages<c64>) {
    if !a.is_square() {
        return (Err(Error::BadIn), LUStages::<c64>::new_with_complex());
    }
    if a.nrows() != _b.len() {
        return (Err(Error::BadIn), LUStages::<c64>::new_with_complex());
    }
    if a.det().unwrap() == 0. {
        return (Err(Error::MultipleSolution), LUStages::<c64>::new_with_complex());
    }
    let b = Array1::<Complex<f64>>::from_shape_fn(_b.len(),
                                                  |i| Complex::from_f64(_b[i]).unwrap());
    let (l, u, stages) =
        match direct_factorization_with_complex(&a) {
            (Ok((_l, _u)), _stages) => (_l, _u, _stages),
            (Err(e), _stages) => return (Err(e), _stages),
        };
    let z = match forward_substitution(&l, &b) {
        Ok(_z) => _z,
        Err(e) => return (Err(e), stages),
    };
    let x_ = match back_substitution(&u, &z) {
        Ok(_x) => _x,
        Err(e) => return (Err(e), stages),
    };
    (Ok(Array1::<f64>::from_shape_fn(b.len(), |i| x_[i].re)), stages)
}

/*pub fn jacobi(a: &Array2<f64>, b: &Array1<f64>, _x0: &Array1<f64>, _tol: f64, max_it: usize) -> Result<(Array1<f64>, usize), Error> {
    let n = b.len();
    let tol = _tol.abs();
    let mut x0 = _x0.clone();
    println!("0 --- {}", x0);
    let (mut err, mut i) = (f64::infinity(), 0usize);
    let mut x1 = Array1::<f64>::zeros(n);
    while err > tol && i < max_it {
        x1 = new_jacobi_set(&a, &b, &x0)?;
        println!("{} --- {}", i+1, x1);
        err = (&x1 - &x0).norm();
        x0.clone_from(&x1);
        i += 1;
    }
    Ok((x1, i))
}

fn new_jacobi_set(a: &Array2<f64>, b: &Array1<f64>,x: &Array1<f64>) -> Result<Array1<f64>, Error> {
    let n = b.len();
    let mut xn = Array1::<f64>::zeros(n);
    let mut sum: f64;
    for i in 0..n {
        if a[(i, i)].is_zero() {
            return Err(Error::DivBy0);
        }
        sum = Zip::from(a.slice(s![i, ..])).and(x.slice(s![..]))
            .fold(0., |ac, a_ij, x_j| ac + a_ij * x_j) - a[(i, i)];
        xn[i] = (b[i] - sum) / a[(i, i)];
    }
    Ok(xn)
}

pub fn gauss_seidel(a: &Array2<f64>, b: &Array1<f64>, _x0: &Array1<f64>, w: f64, _tol: f64, max_it: usize) -> Result<(Array1<f64>, usize), Error> {
    let n = b.len();
    let tol = _tol.abs();
    let mut x0 = _x0.clone();
    println!("0 --- {}", x0);
    let (mut err, mut i) = (f64::infinity(), 0usize);
    let mut x1 = Array1::<f64>::zeros(n);
    while err > tol && i < max_it {
        x1 = new_gauss_set(&a, &b, &x0, w)?;
        err = (&x1 - &x0).norm_l2();
        println!("{} --- {} --- {}", i+1, x1, err);
        x0.clone_from(&x1);
        i += 1;
    }
    Ok((x1, i))
}

fn new_gauss_set(a: &Array2<f64>, b: &Array1<f64>,x: &Array1<f64>, w: f64) -> Result<Array1<f64>, Error> {
    let n = b.len();
    let mut xn = Array1::<f64>::zeros(n);
    let mut sum: f64;
    for i in 0..n {
        if a[(i, i)].is_zero() {
            return Err(Error::DivBy0);
        }
        sum = Zip::from(a.slice(s![i, ..i])).and(xn.slice(s![..i]))
            .fold(0., |ac, a_ij, x_j| ac + a_ij * x_j);
        sum = Zip::from(a.slice(s![i, i+1..])).and(x.slice(s![i+1..]))
            .fold(sum, |ac, a_ij, x_j| ac + a_ij * x_j);
        xn[i] = (1. - w) * x[i] - w * (b[i] + sum) / a[(i, i)];
    }
    Ok(xn)
}*/

pub fn iterate(a: &Array2<f64>, b: &Array1<f64>, _x0: &Array1<f64>, method: IterationType, _tol: f64, max_it: usize) -> (Result<(Array1<f64>, f64), Error>, Table) {
    let mut table = Table::new();
    let n = b.len();
    if !a.is_square() || n != a.nrows() {
        return (Err(Error::BadIn), table);
    }
    let diag_a = a.diag();
    if diag_a.iter().find(|e| **e == 0.) != None {
        return (Err(Error::BadIn), table);
    }
    let tol = _tol.abs();
    let mut x_n = _x0.clone();
    let mut x_n1 = Array1::<f64>::zeros(n);
    let d = Array2::<f64>::from_diag(&a.diag());
    let u = Array2::<f64>::from_shape_fn((n, n),
                                         |(i, j)| {
        if i < j {
            return -a[(i, j)]
        } 0.
    });
    let l = Array2::<f64>::from_shape_fn((n, n),
                                         |(i, j)| {
        if i > j {
            return -a[(i, j)]
        } 0.
    });
    let (t, c) = match method {
        IterationType::Jacobi => {
            let d_inv = d.inv().unwrap();
            (d_inv.dot(&(&l + &u)), d_inv.dot(b)) },
        IterationType::GaussSeidel => {
            let dl_inv = (&d - &l).inv().unwrap();
            (dl_inv.dot(&u), dl_inv.dot(b)) },
        IterationType::SOR(w) => {
            let dl_inv = (&(&d - &(&l * w))).inv().unwrap();
            (dl_inv.dot(&(&(&d * (1. - w)) + &(&u * w))), dl_inv.dot(b) * w) },
    };
    table.set_t_c(&t, &c);
    let mut i = 0usize;
    let mut err = f64::infinity();
    let spec = match spectral_radius(&t) {
        Ok(e) => e,
        Err(_) => return (Err(Error::BadIn), table)
    };
    println!("spectral radius of T: {}", spec);
    println!("{:^4} | {:^4.2E} | {}", i, err, x_n);
    while err > tol && i < max_it {
        x_n1 = &t.dot(&x_n) + &c;
        err = (&x_n1 - &x_n).norm();
        x_n.clone_from(&x_n1);
        i += 1;
        println!("{:^4} | {:^4.2E} | {}", i, err, x_n1);
        table.registry(i, err, &x_n1)
    }
    if err <= tol {
        return (Ok((x_n1, spec)), table)
    }
    (Err(Error::NotEnoughIterations), table)
}



