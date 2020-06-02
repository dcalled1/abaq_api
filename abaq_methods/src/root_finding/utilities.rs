use num_traits::{abs, Float};

#[derive(Debug, Copy, Clone)]
pub enum Pessimistic {
    MaxIterationsReached,
    DivBy0,
    FunctionOutOfDomain,
    ComplexRoot,
    MultipleRoot,
    InvalidInput,
    InvalidFunction,
}

#[derive(Debug, Copy, Clone)]
pub enum Optimistic {
    RootFound,
    RootApproxFound,
    IntervalFound,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Error {
    Absolute,
    Relative
}

pub(crate) fn calc_error(x_prev: f64, x_act: f64, error_type: Error) -> f64 {
    if error_type == Error::Relative && abs(x_act) > Float::epsilon() {
        abs((x_act - x_prev)/x_act)
    } else {
        abs(x_act - x_prev)
    }
}

pub fn load_function(f: String) -> Result<impl Fn(f64) -> f64, Pessimistic>{
    let expr: meval::Expr = match f.as_str().parse() {
        Ok(e) => e,
        Err(_) => return Err(Pessimistic::InvalidFunction),
    };
    match expr.bind("x") {
        Ok(func) => return Ok(func),
        Err(_) => return Err(Pessimistic::InvalidFunction),
    }
}

pub(crate) fn check(n: f64) -> Result<(), Pessimistic> {
    if n.is_infinite() || n.is_nan() {
        return Err(Pessimistic::FunctionOutOfDomain);
    }
    Ok(())
}
