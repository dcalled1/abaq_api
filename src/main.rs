extern crate abaq_methods;
use actix_web::{post, get, web, App, HttpResponse, HttpServer, Responder};
use serde::{Serialize, Deserialize};
use ndarray::{Array2, Array1};
use ndarray::prelude::*;


#[derive(Serialize, Deserialize, Debug)]
enum ErrorType {
    Absolute,
    Relative,
}

#[derive(Serialize, Deserialize, Debug)]
enum Norm {
    Infinite,
    L1,
    L2,
}

#[derive(Serialize, Deserialize, Debug)]
enum ProblemType {
    RootFinding,
    LinearEquations,
    Interpolation,
}

#[derive(Serialize, Deserialize, Debug)]
enum Method {
    IncrementalSearch,
    Bisection,
    FalsePosition,
    FixedPoint,
    Newton,
    Secant,
    MultipleRoot,
    Steffensen,
    Muller,
    AcceleratedFixedPoint,

    GaussianElimination,
    PartialPivotingGaussianElimination,
    TotalPivotingGaussianElimination,
    GaussianFactorization,
    PivotingGaussianFactorization,
    Cholesky,
    Crout,
    Doolittle,
    GaussSeidel,
    Jacobi,
    SOR,

    Vandermonde,
    DividedDifferences,
    Lagrange,
    LinearSplines,
    QuadraticSplines,
    CubicSplines
}

#[derive(Serialize, Deserialize, Debug)]
struct RequestTest {
    problem_type: ProblemType,
    method: Method,
    f: Option<String>,
    g: Option<String>,
    df: Option<String>,
    d2f: Option<String>,
    x0: Option<f64>,
    xa: Option<f64>,
    x1: Option<f64>,
    x2: Option<f64>,
    tol: Option<f64>,
    n: Option<usize>,
    err: Option<ErrorType>,
    a: Option<Vec<f64>>,
    b: Option<Vec<f64>>,
    norm: Option<Norm>,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,



}

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .service(index)
            .service(api)
    })
        .bind("127.0.0.1:8088")?
        .run()
        .await
}

#[get("/")]
async fn index() -> impl Responder {
    format!("Hello :)")
}

#[post("/api")]
async fn api(json: web::Json<RequestTest>) -> impl Responder {
    match json.problem_type {
        ProblemType::RootFinding => {
            match json.method {
                Method::IncrementalSearch => {},
                Method::Bisection => {},
                Method::FalsePosition => {},
                Method::FixedPoint => {},
                Method::Newton => {},
                Method::Secant => {},
                Method::MultipleRoot => {},
                Method::Steffensen => {},
                Method::Muller => {},
                Method::AcceleratedFixedPoint => {},
                _ => (),
            }
        },
        ProblemType::LinearEquations => {
            match json.method {
                Method::GaussianElimination => {},
                Method::PartialPivotingGaussianElimination => {},
                Method::TotalPivotingGaussianElimination => {},
                Method::GaussianFactorization => {},
                Method::PivotingGaussianFactorization => {},
                Method::Cholesky => {},
                Method::Crout => {},
                Method::Doolittle => {},
                Method::GaussSeidel => {},
                Method::Jacobi => {},
                Method::SOR => {},
                _ => (),
            }
        },
        ProblemType::Interpolation => {
            match json.method {
                Method::Vandermonde => {},
                Method::DividedDifferences => {},
                Method::Lagrange => {},
                Method::LinearSplines => {},
                Method::QuadraticSplines => {},
                Method::CubicSplines => {},
                _ => (),
            }
        },
    }

    json
}

