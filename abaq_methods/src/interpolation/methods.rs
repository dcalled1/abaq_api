use ndarray::{Array1, Array2, Zip};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use crate::interpolation::utilities::{Error, Splines};
use ndarray_linalg::{Solve, Determinant};

pub fn vandermonde(x: &Array1<f64>, y: &Array1<f64>) -> Result<String, Error>{
    let n = x.len();
    if n != y.len() {
        return Err(Error::BadIn);
    }
    let m = Array2::<f64>::from_shape_fn((n, n),
        |(i, j)| {
            x[i].powi((n - j - 1) as i32)
        }
    );
    let de = m.det().unwrap();
    println!("det: {}\n{}", de, m);
    let a = m.solve_into(y.clone()).unwrap();
    let mut pol = String::new();
    for i in (1..n).rev() {
        let e = a[n - i - 1];
        if e != 0. {
            let xi = match i {
                0 => String::new(),
                1 => String::from("x"),
                _ => format!("x^{}", i),
            };
            if e == 1. {
                pol.push_str(format!("{}", xi).as_str())
            } else if e == -1. {
                pol.push_str(format!("-{}", xi).as_str())
            } else {
            pol.push_str(format!("{:+}*{}", e, xi).as_str())
        }
        }
    }
    if a[n - 1] != 0. {
        pol.push_str(format!("{:+}", a[n - 1]).as_str());
    }
    if pol.starts_with("+") {
        pol.remove(0);
    }
    println!("{}\n{}", a, pol);
    Ok(pol)
}

pub fn divided_differences(x: &Array1<f64>, fx: &Array1<f64>) -> Result<String, Error> {
    let n = x.len();
    if n != fx.len() {
        return Err(Error::BadIn);
    }
    let mut m = Array2::<f64>::zeros((n, n + 1));
    let (mut c0, mut c1) = m.multi_slice_mut((s![.., 0], s![.., 1]));
    Zip::from(&mut c0).and(&mut c1).and(x).and(fx).par_apply(|m0, m1, xi, yi| {
        *m0 = *xi;
        *m1 = *yi;
    });
    //println!("{}", m);

    for k in 2..=n {
        for i in k - 1..n {
            m[(i, k)] = (m[(i, k-1)] - m[(i - 1, k - 1)]) / (x[i] - x[i + 1 - k]);
        }
    }
    println!("{}", m);
    let mut pol = format!("{:+}", m[(0, 1)]);
    for i in 1..n {
        if m[(i, i + 1)] != 0. {
            pol.push_str(format!("{:+}", m[(i, i + 1)]).as_str());
            for j in 0..i {

                pol.push_str(format!("*(x{:+})", -x[j]).as_str());
            }
        }
    }
    if pol.starts_with("+") {
        pol.remove(0);
    }
    println!("{}", pol);
    Ok(pol)
}

pub fn lagrange_pol(x: &Array1<f64>, y: &Array1<f64>) -> Result<String, Error> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::BadIn);
    }
    let mut pol = String::new();
    let ls = Zip::indexed(x).par_apply_collect(|i, xi| {
        x.indexed_iter().filter(|(j, _)| {
            *j != i
        }).fold(1., |acc, (_, xj)| {
            acc * (xi - xj)
        })
    });
    for i in 0..n {
        pol.push_str(format!("{:+}", y[i] / ls[i]).as_str());
        for j in 0..i {
            pol.push_str(format!("(x{:+})", -x[j]).as_str());
        }
        for j in i+1..n {
            pol.push_str(format!("(x{:+})", -x[j]).as_str());
        }
    }
    if pol.starts_with("+") {
        pol.remove(0);
    }
    println!("\n\n{}", pol);
    Ok(pol)
}

pub fn linear_splines(x: &Array1<f64>, y: &Array1<f64>) -> Result<Splines, Error> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::BadIn);
    }
    let (mut a, mut x1, mut x2, mut y1): (f64, f64, f64, f64);
    let mut pol: String;
    let mut splines = Splines::new();
    for i in 1..n {
        x1 = x[i - 1];
        x2 = x[i];
        if x1 >= x2 {
            return Err(Error::BadIn);
        }
        y1 = y[i - 1];
        a = (y[i] - y1) / (x2 - x1);
        if a == 1. {
            pol = String::from("x");
        } else {
            pol = format!("{}*x", a)
        }

        if y1 != a * x1 {
            pol.push_str(format!("{:+}", y1 - a * x1).as_str());
        }
        splines.add(x1, x2, &pol);
    }
    Ok(splines)
}

pub fn quadratic_splines(x: &Array1<f64>, y: &Array1<f64>) -> Result<(Splines, Array2<f64>), Error> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::BadIn);
    }
    let mut splines = Splines::new();
    let a_size = 3 * (n - 1);
    let mut a = Array2::<f64>::zeros((a_size, a_size));
    a[(0, 0)] = x[0].powi(2);
    a[(0, 1)] = x[0];
    a[(0, 2)] = 1.;
    a[(a_size - 1, 0)] = 2.;

    let (mut i_1, mut i_n, mut i_3, mut i_2n): (usize, usize, usize, usize);
    for i in 0..n-1 {
        i_1 = i + 1;
        i_3 = i * 3;
        a[(i_1, i_3    )] = x[i_1].powi(2);
        a[(i_1, i_3 + 1)] = x[i_1];
        a[(i_1, i_3 + 2)] = 1.;
    }
    for i in 0..n-2 {
        i_n = i + n;
        i_1 = i + 1;
        i_3 = i * 3;
        a[(i_n, i_3    )] = x[i_1].powi(2);
        a[(i_n, i_3 + 1)] = x[i_1];
        a[(i_n, i_3 + 2)] = 1.;
        a[(i_n, i_3 + 3)] = -x[i_1].powi(2);
        a[(i_n, i_3 + 4)] = -x[i_1];
        a[(i_n, i_3 + 5)] = -1.;
    }

    for i in 0..n-2 {
        i_2n = i + 2 * n - 2;
        i_1 = i + 1;
        i_3 = i * 3;
        a[(i_2n, i_3    )] = 2. * x[i_1];
        a[(i_2n, i_3 + 1)] = 1.;

        a[(i_2n, i_3 + 3)] = -2. * x[i_1];
        a[(i_2n, i_3 + 4)] = -1.;
    }
    let mut b = Array1::<f64>::zeros(a_size);
    Zip::from(b.slice_mut(s![..n])).and(y).par_apply(|bi, yi| *bi = *yi);
    let ans = a.solve_into(b).unwrap();

    let mut pol: String;
    let (mut ai, mut bi, mut ci): (f64, f64, f64);
    for i in 1..n {
        ai = ans[(i - 1) * 3    ];
        bi = ans[(i - 1) * 3 + 1];
        ci = ans[(i - 1) * 3 + 2];
        pol = match ai {
            0.  => String::new(),
            1.  => String::from("x^2"),
            -1. => String::from("-x^2"),
            _   => format!("{}*x^2", ai),
        };
        match bi {
            0.  => (),
            1.  => pol.push('x'),
            -1. => pol.push_str("-x"),
            _   => pol.push_str(format!("{:+}*x", bi).as_str()),
        }
        match ci {
            0. => (),
            _  => pol.push_str(format!("{:+}", ci).as_str()),
        }
        if pol.starts_with("+") {
            pol.remove(0);
        }

        splines.add(x[i-1], x[i], &pol);
    }


    println!("{}\n\n{}", a, ans);
    Ok((splines, a))
}

pub fn cubic_splines(x: &Array1<f64>, y: &Array1<f64>) -> Result<(Splines, Array2<f64>), Error> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::BadIn);
    }
    let mut splines = Splines::new();
    let a_size = 4 * (n -1);
    let mut a = Array2::<f64>::zeros((a_size, a_size));
    a[(0, 0)] = x[0].powi(3);
    a[(0, 1)] = x[0].powi(2);
    a[(0, 2)] = x[0];
    a[(0, 3)] = 1.;

    a[(a_size - 2, 0)] = 6. * x[0];
    a[(a_size - 2, 1)] = 2.;

    a[(a_size - 1, a_size - 4)] = 6. * x[n - 1];
    a[(a_size - 1, a_size - 3)] = 2.;
    let (mut i_1, mut i_n, mut i_4, mut i_2n, mut i_3n): (usize, usize, usize, usize, usize);
    for i in 0..n-1 {
        i_1 = i + 1;
        i_4 = i * 4;
        a[(i_1, i_4    )] = x[i_1].powi(3);
        a[(i_1, i_4 + 1)] = x[i_1].powi(2);
        a[(i_1, i_4 + 2)] = x[i_1];
        a[(i_1, i_4 + 3)] = 1.;
    }
    for i in 0..n-2 {
        i_n = i + n;
        i_1 = i + 1;
        i_4 = i * 4;
        a[(i_n, i_4    )] = x[i_1].powi(3);
        a[(i_n, i_4 + 1)] = x[i_1].powi(2);
        a[(i_n, i_4 + 2)] = x[i_1];
        a[(i_n, i_4 + 3)] = 1.;
        a[(i_n, i_4 + 4)] = -x[i_1].powi(3);
        a[(i_n, i_4 + 5)] = -x[i_1].powi(2);
        a[(i_n, i_4 + 6)] = -x[i_1];
        a[(i_n, i_4 + 7)] = -1.;
    }

    for i in 0..n-2 {
        i_2n = i + 2 * n - 2;
        i_1 = i + 1;
        i_4 = i * 4;
        a[(i_2n, i_4    )] = 3. * x[i_1].powi(2);
        a[(i_2n, i_4 + 1)] = 2. * x[i_1];
        a[(i_2n, i_4 + 2)] = 1.;

        a[(i_2n, i_4 + 4)] = -3. * x[i_1].powi(2);
        a[(i_2n, i_4 + 5)] = -2. * x[i_1];
        a[(i_2n, i_4 + 6)] = -1.;
    }

    for i in 0..n-2 {
        i_3n = i + 3 * n - 4;
        i_1 = i + 1;
        i_4 = i * 4;
        a[(i_3n, i_4    )] = 6. * x[i_1];
        a[(i_3n, i_4 + 1)] = 2.;

        a[(i_3n, i_4 + 4)] = -6. * x[i_1];
        a[(i_3n, i_4 + 5)] = -2.;
    }
    let mut b = Array1::<f64>::zeros(a_size);
    Zip::from(b.slice_mut(s![..n])).and(y).par_apply(|bi, yi| *bi = *yi);
    let ans = a.solve_into(b).unwrap();

    let mut pol: String;
    let (mut ai, mut bi, mut ci, mut di): (f64, f64, f64, f64);
    for i in 1..n {
        ai = ans[(i - 1) * 4    ];
        bi = ans[(i - 1) * 4 + 1];
        ci = ans[(i - 1) * 4 + 2];
        di = ans[(i - 1) * 4 + 3];
        pol = match ai {
            0.  => String::new(),
            1.  => String::from("x^3"),
            -1. => String::from("-x^3"),
            _   => format!("{}*x^3", ai),
        };
        match bi {
            0.  => (),
            1.  => pol.push_str("x^2"),
            -1. => pol.push_str("-x^2"),
            _   => pol.push_str(format!("{:+}*x^2", bi).as_str()),
        }
        match ci {
            0.  => (),
            1.  => pol.push('x'),
            -1. => pol.push_str("-x"),
            _   => pol.push_str(format!("{:+}*x", ci).as_str()),
        }
        match di {
            0. => (),
            _  => pol.push_str(format!("{:+}", di).as_str()),
        }
        if pol.starts_with("+") {
            pol.remove(0);
        }

        splines.add(x[i-1], x[i], &pol);
    }


    println!("{}\n\n{}", a, ans);
    Ok((splines, a))
}
