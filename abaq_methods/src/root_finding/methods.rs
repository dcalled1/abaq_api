use num_traits::{Float, abs};
use crate::root_finding::register::Logbook;
use crate::root_finding::utilities::{Optimistic, Pessimistic, Error, calc_error, check};


/*
pub trait RunnableMethod {
    fn run(&mut self) -> Logbook;
    fn run_with_no_log(&mut self);
    fn next(&mut self) -> Option<f64>;
    fn pursue(&mut self) -> bool;
}*/

//Methods

pub fn incremental_search(f: impl Fn(f64)->f64, x0: f64, dx: f64, n:u32)
    -> Result<(Option<f64>, Option<(f64, f64)>, Optimistic), Pessimistic> {
    let (mut xa, mut xb) = (x0, x0 + dx);
    let (mut ya, mut yb) = (f(xa), f(xb));

    let mut i = 0u32;
    while ya * yb > 0f64 && i < n {
        xa = xb;
        xb = xa + dx;
        ya = f(xa);
        check(ya)?;
        yb = f(xb);
        check(yb)?;
        i += 1;
    }
    if ya == 0f64 {
        return Ok((Some(xa), None, Optimistic::RootFound));
    }
    if yb == 0f64 {
        return Ok((Some(xb), None, Optimistic::RootFound));
    }
    if ya * yb < 0f64 {
        return Ok((None, Some((xa, xb)), Optimistic::IntervalFound));
    }
    Err(Pessimistic::MaxIterationsReached)
}

pub fn bisection(f: impl Fn(f64)->f64, _xu: f64, _xl: f64, tol: f64, n:u32, error_type: Error)
                          -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xl".into(), "xm".into(), "xu".into(),
                                           "yl".into(), "ym".into(), "yu".into(), "error".into()]);
    let (mut xu, mut xl) = if _xu > _xl {
        (_xu, _xl)
    } else {
        (_xl, _xu)
    };
    let mut yu = f(xu);
    match check(yu) {
        Err(e) => return (Err(e), logbook),
        _ => (),
    }
    let mut yl = f(xl);
    match check(yl) {
        Err(e) => return (Err(e), logbook),
        _ => (),
    }
    if yu == 0f64 {
        return (Ok((xu, 0, Optimistic::RootFound)), logbook);
    }
    if yl == 0f64 {
        return (Ok((xl, 0, Optimistic::RootFound)), logbook);
    }
    if yu*yl > 0f64 {
        return (Err(Pessimistic::InvalidInput), logbook);
    }
    let mut xm = (xl + xu)/2f64;
    let mut ym = f(xm);
    match check(ym) {
        Err(e) => return (Err(e), logbook),
        _ => (),
    }
    let mut err: f64 = Float::infinity();
    let mut i = 1u32;
    logbook.registry(i, vec![xl, xm, xu, yl, ym, yu, err]);
    let mut xaux;
    while ym != 0f64 && err > tol && i < n {
        if yl * ym < 0f64 {
            xu = xm;
            yu = ym;
        } else {
            xl = xm;
            yl = ym;
        }
        xaux = xm;
        xm = (xl + xu)/2f64;
        ym = f(xm);
        match check(ym) {
            Err(e) => return (Err(e), logbook),
            _ => (),
        }
        err = calc_error(xm, xaux, error_type);
        i += 1;
        logbook.registry(i, vec![xl, xm, xu, yl, ym, yu, err]);
    }
    if ym == 0f64 {
        return (Ok((xm, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xm, i, Optimistic::RootApproxFound)), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)

}

pub fn false_position(f: impl Fn(f64)->f64, _xu: f64, _xl: f64,
                      tol: f64, n:u32, error_type: Error)
                 -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xl".into(), "xm".into(), "xu".into(),
                                           "yl".into(), "ym".into(), "yu".into(), "error".into()]);
    let (mut xu, mut xl) = if _xu > _xl {
        (_xu, _xl)
    } else {
        (_xl, _xu)
    };
    let mut yu = f(xu);
    match check(yu) {
        Err(e) => return (Err(e), logbook),
        _ => (),
    }
    let mut yl = f(xl);
    match check(yl) {
        Err(e) => return (Err(e), logbook),
        _ => (),
    }
    if yu == 0f64 {
        return (Ok((xu, 0, Optimistic::RootFound)), logbook);
    }
    if yl == 0f64 {
        return (Ok((xl, 0, Optimistic::RootFound)), logbook);
    }
    if yu*yl > 0f64 {
        return (Err(Pessimistic::InvalidInput), logbook);
    }
    let mut xm = xl - yl * (xu - xl)/(yu - yl);
    let mut ym = f(xm);
    match check(ym) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut err: f64 = Float::infinity();
    let mut i = 1u32;
    logbook.registry(i, vec![xl, xm, xu, yl, ym, yu, err]);
    let mut xaux;
    while ym != 0f64 && err > tol && i < n {
        if yl * ym < 0f64 {
            xu = xm;
            yu = ym;
        } else {
            xl = xm;
            yl = ym;
        }
        xaux = xm;
        xm = xl - yl * (xu - xl)/(yu - yl);
        ym = f(xm);
        match check(ym) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        err = calc_error(xaux, xm, error_type);
        i += 1;
        logbook.registry(i, vec![xl, xm, xu, yl, ym, yu, err]);
    }
    if ym == 0f64 {
        return (Ok((xm, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xm, i, Optimistic::RootApproxFound)), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)

}

pub fn fixed_point(f: impl Fn(f64)->f64, g: impl Fn(f64)->f64,
                   _xa: f64,  tol: f64, n:u32, error_type: Error)
                      -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "error".into()]);
    let mut xa = _xa;
    let mut y = f(xa);
    match check(y) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut err: f64 = Float::infinity();
    let mut i = 0u32;
    logbook.registry(i, vec![xa, y, err]);
    let mut xn;
    while y != 0f64 && err > tol && i < n {
        xn = g(xa);
        match check(xn) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        y = f(xn);
        match check(y) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        err = calc_error(xa, xn, error_type);
        xa = xn;
        i += 1;
        logbook.registry(i, vec![xa, y, err]);
    }
    if y == 0f64 {
        return (Ok((xa, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xa, i, Optimistic::RootApproxFound)), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)

}

pub fn newton(f: impl Fn(f64)->f64, df: impl Fn(f64)->f64,
              _xa: f64,  tol: f64, n:u32, error_type: Error)
                   -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "y\'i".into(), "error".into()]);
    let mut xa = _xa;
    let mut y = f(xa);
    match check(y) {
        Err(e) => return (Err(e), logbook),
        _ => (),
    }
    let mut dy = df(xa);
    match check(dy) {
        Err(e) => return (Err(e), logbook),
        _ => (),
    }
    let mut err: f64 = Float::infinity();
    let mut i = 0u32;
    logbook.registry(i, vec![xa, y, dy, err]);
    let mut xn;
    while y != 0f64 && dy != 0f64 && err > tol && i < n {
        xn = xa - y/dy;
        y = f(xn);
        match check(y) {
            Err(e) => return (Err(e), logbook),
            _ => (),
        }
        dy = df(xn);
        match check(dy) {
            Err(e) => return (Err(e), logbook),
            _ => (),
        }
        err = calc_error(xa, xn, error_type);
        xa = xn;
        i += 1;
        logbook.registry(i, vec![xa, y, dy, err]);
    }
    if y == 0f64 {
        return (Ok((xa, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xa, i, Optimistic::RootApproxFound)), logbook);
    }
    if dy == 0f64 {
        return (Err(Pessimistic::MultipleRoot), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}

pub fn secant(f: impl Fn(f64)->f64, _x0: f64, _x1: f64,  tol: f64,
              n:u32, error_type: Error)
              -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "error".into()]);
    let (mut x0, mut x1) = (_x0, _x1);
    let mut y0 = f(x0);
    match check(y0) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    if y0 == 0f64 {
        return (Ok((x0, 0, Optimistic::RootFound)), logbook)
    }
    let mut y1 = f(x1);
    match check(y1) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut err: f64 = Float::infinity();
    let mut i = 1u32;
    let mut x2;
    logbook.registry(0, vec![x0, y0, err]);
    logbook.registry(1, vec![x1, y1, err]);
    while y1 != 0f64 && y1 != y0 && err > tol && i < n {
        x2 = x1 - y1 * (x1 - x0)/(y1 - y0);
        err = calc_error(x1, x2, error_type);
        x0 = x1;
        y0 = y1;
        x1 = x2;
        y1 = f(x1);
        match check(y1) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        i += 1;
        logbook.registry(i, vec![x1, y1, err]);
    }
    if y1 == 0f64 {
        return (Ok((x1, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((x1, i, Optimistic::RootApproxFound)), logbook);
    }
    if y0 == y1 {
        return (Err(Pessimistic::MultipleRoot), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}


pub fn multiple_root(f: impl Fn(f64)->f64, df: impl Fn(f64)->f64,
                     d2f: impl Fn(f64)->f64,_xa: f64,  tol: f64,
                     n:u32, error_type: Error)
                     -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "y\'i".into(),
                                           "y\'\'i".into(), "error".into()]);
    let mut xa = _xa;
    let mut y = f(xa);
    match check(y) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut dy = df(xa);
    match check(dy) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut d2y = d2f(xa);
    match check(d2y) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut err: f64 = Float::infinity();
    let mut i = 0u32;
    logbook.registry(i, vec![xa, y, dy, d2y, err]);
    let mut xn;
    while y != 0f64 && dy.powi(2) != y * d2y && err > tol && i < n {
        xn = xa - y * dy /(dy.powi(2) - y * d2y);
        y = f(xn);
        match check(y) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        dy = df(xn);
        match check(dy) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        d2y = d2f(xn);
        match check(d2y) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        err = if error_type == Error::Relative && abs(xn) > Float::epsilon() {
            abs((xn - xa)/xn)
        } else {
            abs(xn - xa)
        };
        xa = xn;
        i += 1;
        logbook.registry(i, vec![xa, y, dy, d2y, err]);
    }
    if y == 0f64 {
        return (Ok((xa, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xa, i, Optimistic::RootApproxFound)), logbook);
    }
    if dy.powi(2) == y * d2y {
        return (Err(Pessimistic::DivBy0), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}


pub fn steffensen(f: impl Fn(f64)->f64, _xa: f64,  tol: f64, n:u32, error_type: Error)
              -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "z\'i".into(),
                                           "error".into()]);
    let mut xa = _xa;
    let mut y = f(xa);
    match check(y) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut z = f(xa + y);
    match check(z) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut err: f64 = Float::infinity();
    let mut i = 0u32;
    logbook.registry(i, vec![xa, y, z, err]);
    let mut xn;
    while y != 0f64 && y != z && err > tol && i < n {
        xn = xa - y.powi(2)/(z - y);
        y = f(xn);
        match check(y) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        z = f(xn + y);
        match check(z) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        err = calc_error(xa, xn, error_type);
        xa = xn;
        i += 1;
        logbook.registry(i, vec![xa, y, z, err]);
    }
    if y == 0f64 {
        return (Ok((xa, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xa, i, Optimistic::RootApproxFound)), logbook);
    }
    if y == z {
        return (Err(Pessimistic::MultipleRoot), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}


pub fn muller(f: impl Fn(f64)->f64, _x0: f64, _x1: f64, _x2: f64,
              tol: f64, n:u32, error_type: Error)
              -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "error".into()]);
    let (mut x0, mut x1, mut x2) = (_x0, _x1, _x2);
    let mut y0 = f(x0);
    match check(y0) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut err: f64 = Float::infinity();
    logbook.registry(0, vec![x0, y0, err]);
    if y0 == 0f64 {
        return (Ok((x0, 0, Optimistic::RootFound)), logbook)
    }
    let mut y1 = f(x1);
    match check(y1) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    err = calc_error(x0, x1, error_type);
    logbook.registry(1, vec![x1, y1, err]);
    if y1 == 0f64 {
        return (Ok((x1, 0, Optimistic::RootFound)), logbook)
    }
    let mut y2 = f(x2);
    match check(y2) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    err = calc_error(x1, x2, error_type);
    logbook.registry(2, vec![x2, y2, err]);
    let (mut h0, mut h1) = (x1 - x0, x2 - x1);
    if h0 == 0f64 || h1 == 0f64 {
        return (Err(Pessimistic::DivBy0), logbook);
    }
    let (mut d0, mut d1) = ((y1 - y0)/h0, (y2 - y2)/h1);
    if abs(h1 + h0) < Float::epsilon() {
        return (Err(Pessimistic::DivBy0), logbook);
    }
    let mut a = (d1 - d0)/(h1 + h0);
    let mut b = a * h1 + d1;
    let mut i = 2u32;
    let mut x3;
    println!("{} - {}", a, b);
    while y2 != 0f64 && b.powi(2) >= 4f64 * a * y2
        && x0 != x1 && x1 != x2 && x0 != x2 && err > tol && i < n {
        x3 = if b < 0f64 {
            x2 + 2f64 * y2 / (b - (b.powi(2) - 4f64 * a * y2).sqrt())
        } else {
            x2 + 2f64 * y2 / (b + (b.powi(2) - 4f64 * a * y2).sqrt())
        };
        err = calc_error(x2, x3, error_type);
        x0 = x1;
        y0 = y1;
        x1 = x2;
        y1 = y2;
        x2 = x3;
        y2 = f(x2);
        match check(y2) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        h0 = x1 - x0;
        h1 = x2 - x1;
        i += 1;
        logbook.registry(i, vec![x2, y2, err]);
        if h0 == 0f64 || h1 == 0f64 {
            return (Err(Pessimistic::DivBy0), logbook);
        }
        d0 = (y1 - y0)/h0;
        d1 = (y2 - y2)/h1;
        if abs(h1 + h0) < Float::epsilon() {
            return (Err(Pessimistic::DivBy0), logbook);
        }
        a = (d1 - d0)/(h1 + h0);
        b = a * h1 + d1;
    }
    if y1 == 0f64 {
        return (Ok((x2, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((x2, i, Optimistic::RootApproxFound)), logbook);
    }
    if x0 == x1 || x1 == x2 || x0 == x2 {
        return (Err(Pessimistic::MultipleRoot), logbook);
    }
    if b.powi(2) < 4f64 * a * y2 {
        return (Err(Pessimistic::ComplexRoot), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}

pub fn accelerated_fixed_point(f: impl Fn(f64)->f64, g: impl Fn(f64)->f64, _x0: f64,
                               tol: f64, n:u32, error_type: Error)
                                -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "error".into()]);
    let mut x0 = _x0;
    let mut x1 = g(x0);
    match check(x1) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut x2 = g(x1);
    match check(x2) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    match check(f(x0)) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    logbook.registry(0, vec![x0, f(x0), Float::infinity()]);
    match check(f(x1)) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    logbook.registry(1, vec![x1, f(x1), calc_error(x0, x1, error_type)]);
    match check(f(x2)) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    logbook.registry(2, vec![x2, f(x2), calc_error(x1, x2, error_type)]);
    if x2 + x0 == 2f64 * x1 {
        return (Err(Pessimistic::DivBy0), logbook);
    }
    let mut x3 = x2 - (x2 - x1).powi(2)/(x2 - 2f64 * x1 + x0);
    let mut y = f(x3);
    match check(y) {
        Err(e) => return (Err(e), logbook),
        Ok(_) => (),
    }
    let mut err = calc_error(x2, x3, error_type);
    let mut i = 3u32;
    logbook.registry(i, vec![x3, y, err]);
    while y != 0f64 && err > tol && i < n {
        x0 = x3;
        x1 = g(x0);
        match check(x1) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        x2 = g(x1);

        match check(x2) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        if x2 + x0 == 2f64 * x1 {
            return (Err(Pessimistic::DivBy0), logbook);
        }
        x3 = x2 - (x2 - x1).powi(2)/(x2 - 2f64 * x1 + x0);
        y = f(x3);
        match check(y) {
            Err(e) => return (Err(e), logbook),
            Ok(_) => (),
        }
        err = calc_error(x2, x3, error_type);
        i += 1;
        logbook.registry(i, vec![x3, y, err]);
    }
    if y == 0f64 {
        return (Ok((x3, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((x3, i, Optimistic::RootApproxFound)), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)

}
