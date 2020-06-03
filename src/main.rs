extern crate abaq_methods;
use actix_web::{post, get, web, App, HttpResponse, HttpServer, Responder};
use serde::{Serialize, Deserialize};
use ndarray::{Array2, Array1};
use ndarray::prelude::*;
use abaq_methods::root_finding::utilities::*;
use abaq_methods::root_finding::methods::*;
use abaq_methods::root_finding::register::Logbook;
use abaq_methods::linear_equations::methods::*;
use abaq_methods::linear_equations::utilities::{Stages, LUStages, Table, IterationType};
use ndarray_linalg::c64;
use abaq_methods::linear_equations;
use abaq_methods::interpolation::methods::{vandermonde, divided_differences, linear_splines, quadratic_splines, cubic_splines};
use abaq_methods::interpolation::utilities::Splines;


#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
enum ErrorType {
    Absolute,
    Relative,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
enum Norm {
    Infinite,
    L1,
    L2,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
enum ProblemType {
    RootFinding,
    LinearEquations,
    Interpolation,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
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

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RequestTest {
    problem_type: ProblemType,
    method: Method,
    f: Option<String>,
    g: Option<String>,
    df: Option<String>,
    d2f: Option<String>,
    xa: Option<f64>,
    xu: Option<f64>,
    xl: Option<f64>,
    x0: Option<f64>,
    x1: Option<f64>,
    x2: Option<f64>,
    dx: Option<f64>,
    tol: Option<f64>,
    n: Option<usize>,
    err: Option<ErrorType>,
    a: Option<Vec<f64>>,
    b: Option<Vec<f64>>,
    w: Option<f64>,
    norm: Option<Norm>,
    v0: Option<Vec<f64>>,
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

#[derive(Serialize)]
struct IncrementalSearchResponse {
    root: Option<f64>,
    xa: Option<f64>,
    xb: Option<f64>,
}

#[derive(Serialize)]
struct RootFindingResponse {
    root: f64,
    case: Optimistic,
    table: Logbook,
}

#[derive(Serialize)]
struct RootFindingFail {
    err: Pessimistic,
    table: Logbook,
}

#[derive(Serialize)]
struct LinearEquationsResponse {
    x: Array1<f64>,
    stages: Stages,
}

#[derive(Serialize)]
struct LinearEquationsFail {
    err: linear_equations::utilities::Error,
    stages: Stages,
}

#[derive(Serialize)]
struct ComplexLULinearEquationsResponse {
    x: Array1<f64>,
    stages: LUStages<c64>,
}

#[derive(Serialize)]
struct ComplexLULinearEquationsFail {
    err: linear_equations::utilities::Error,
    stages: LUStages<c64>,
}

#[derive(Serialize)]
struct LULinearEquationsFail {
    err: linear_equations::utilities::Error,
    stages: LUStages<f64>,
}

#[derive(Serialize)]
struct LULinearEquationsResponse {
    x: Array1<f64>,
    stages: LUStages<f64>,
}

#[derive(Serialize)]
struct IterativeLULinearEquationsFail {
    err: linear_equations::utilities::Error,
    stages: Table,
}

#[derive(Serialize)]
struct IterativeLULinearEquationsResponse {
    x: Array1<f64>,
    stages: Table,
    spectral_radius: f64,
}

#[derive(Serialize)]
struct SplinesMatrix {
    m: Array2<f64>,
    splines: Splines,
}


#[post("/api")]
async fn api(json: web::Json<RequestTest>) -> impl Responder {
    match json.problem_type {
        ProblemType::RootFinding => {
            match json.method {
                Method::IncrementalSearch => {
                    let f_str = json.f.clone().unwrap();
                    let x0 = json.x0.unwrap();
                    let dx = json.dx.unwrap();
                    let n = json.n.unwrap();
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function"),
                    };
                    match incremental_search(f, x0, dx, n) {
                        Ok((root, interval, ans)) => {
                            match ans {
                                Optimistic::RootFound => return HttpResponse::Ok().json(IncrementalSearchResponse {
                                    root,
                                    xa: None,
                                    xb: None,
                                }),
                                Optimistic::IntervalFound => {
                                    let (xa, xb) = interval.unwrap();
                                    return HttpResponse::Ok().json(IncrementalSearchResponse {
                                        root,
                                        xa: Some(xa),
                                        xb: Some(xb),
                                    });
                                },
                                _ => (),
                            }
                        },
                        Err(e) => return HttpResponse::Ok().json(e),
                    };
                },
                Method::Bisection => {
                    let f_str = json.f.clone().unwrap();
                    let xu = json.xu.unwrap();
                    let xl = json.xl.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function"),
                    };
                    match bisection(f, xu, xl, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            return HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            });
                        },
                        _ => (),
                    };
                },
                Method::FalsePosition => {
                    let f_str = json.f.clone().unwrap();
                    let xu = json.xu.unwrap();
                    let xl = json.xl.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function"),
                    };
                    return match false_position(f, xu, xl, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            })
                        },
                        (Err(e), table) => HttpResponse::Ok().json(RootFindingFail {
                            err: e,
                            table,
                        }),
                    };
                },
                Method::FixedPoint => {
                    let f_str = json.f.clone().unwrap();
                    let g_str = json.g.clone().unwrap();
                    let xa = json.xa.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (f)"),
                    };
                    let g = match load_function(g_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (g)"),
                    };
                    return match fixed_point(f, g, xa, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            })
                        },
                        (Err(e), table) => HttpResponse::Ok().json(RootFindingFail {
                            err: e,
                            table,
                        }),
                    };
                },
                Method::Newton => {
                    let f_str = json.f.clone().unwrap();
                    let df_str = json.df.clone().unwrap();
                    let xa = json.xa.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (f)"),
                    };
                    let df = match load_function(df_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (df)"),
                    };
                    return match newton(f, df, xa, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            })
                        },
                        (Err(e), table) => HttpResponse::Ok().json(RootFindingFail {
                            err: e,
                            table,
                        }),
                    };
                },
                Method::Secant => {
                    let f_str = json.f.clone().unwrap();
                    let x0 = json.x0.unwrap();
                    let x1 = json.x1.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function"),
                    };
                    return match secant(f, x0, x1, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            })
                        },
                        (Err(e), table) => HttpResponse::Ok().json(RootFindingFail {
                            err: e,
                            table,
                        }),
                    };
                },
                Method::MultipleRoot => {
                    let f_str = json.f.clone().unwrap();
                    let df_str = json.f.clone().unwrap();
                    let d2f_str = json.f.clone().unwrap();
                    let xa = json.x0.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (f)"),
                    };
                    let df = match load_function(df_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (df)"),
                    };
                    let d2f = match load_function(d2f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (d2f)"),
                    };

                    return match multiple_root(f, df, d2f,xa, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            })
                        },
                        (Err(e), table) => HttpResponse::Ok().json(RootFindingFail {
                            err: e,
                            table,
                        }),
                    };
                },
                Method::Steffensen => {
                    let f_str = json.f.clone().unwrap();
                    let xa = json.xa.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function"),
                    };
                    return match steffensen(f, xa, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            })
                        },
                        (Err(e), table) => HttpResponse::Ok().json(RootFindingFail {
                            err: e,
                            table,
                        }),
                    };
                },
                Method::Muller => {
                    let f_str = json.f.clone().unwrap();
                    let x0 = json.x0.unwrap();
                    let x1 = json.x1.unwrap();
                    let x2 = json.x2.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (f)"),
                    };
                    return match muller(f, x0, x1, x2, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            })
                        },
                        (Err(e), table) => HttpResponse::Ok().json(RootFindingFail {
                            err: e,
                            table,
                        }),
                    };
                },
                Method::AcceleratedFixedPoint => {
                    let f_str = json.f.clone().unwrap();
                    let g_str = json.g.clone().unwrap();
                    let xa = json.xa.unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let error_type = match json.err.unwrap() {
                        ErrorType::Absolute => Error::Absolute,
                        ErrorType::Relative => Error::Relative,
                    };
                    let f = match load_function(f_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (f)"),
                    };
                    let g = match load_function(g_str) {
                        Ok(_f) => _f,
                        _ => return HttpResponse::BadRequest().body("Invalid Function (g)"),
                    };
                    return match accelerated_fixed_point(f, g, xa, tol, n as u32, error_type) {
                        (Ok((root, it, ans)), table) => {
                            HttpResponse::Ok().json(RootFindingResponse {
                                root,
                                case: ans,
                                table,
                            })
                        },
                        (Err(e), table) => HttpResponse::Ok().json(RootFindingFail {
                            err: e,
                            table,
                        }),
                    };
                },
                _ => return HttpResponse::BadRequest().body("Invalid Method"),
            }
        },
        ProblemType::LinearEquations => {
            match json.method {
                Method::GaussianElimination => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let n = json.n.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match gaussian_elimination(&a, &b) {
                        (Ok(x), stages) => HttpResponse::Ok()
                            .json(LinearEquationsResponse{
                            x,
                            stages
                        }),
                        (Err(e), stages) => HttpResponse::Ok().json(LinearEquationsFail {
                            err: e,
                            stages
                        }),
                    }
                },
                Method::PartialPivotingGaussianElimination => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let n = json.n.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match gaussian_elimination_partial_pivoting(&a, &b) {
                        (Ok(x), stages) => HttpResponse::Ok()
                            .json(LinearEquationsResponse{
                                x,
                                stages
                            }),
                        (Err(e), stages) => HttpResponse::Ok().json(LinearEquationsFail {
                            err: e,
                            stages
                        }),
                    };
                },
                Method::TotalPivotingGaussianElimination => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let n = json.n.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match gaussian_elimination_total_pivoting(&a, &b) {
                        (Ok(x), stages) => HttpResponse::Ok()
                            .json(LinearEquationsResponse{
                                x,
                                stages
                            }),
                        (Err(e), stages) => HttpResponse::Ok().json(LinearEquationsFail {
                            err: e,
                            stages
                        }),
                    };
                },
                Method::GaussianFactorization => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let n = json.n.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match gaussian_factorization(&a, &b) {
                        (Ok(x), stages) => HttpResponse::Ok()
                            .json(LULinearEquationsResponse{
                                x,
                                stages
                            }),
                        (Err(e), stages) => HttpResponse::Ok()
                            .json(LULinearEquationsFail {
                            err: e,
                            stages
                        }),
                    };
                },
                Method::PivotingGaussianFactorization => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let n = json.n.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match pivoting_gaussian_factorization(&a, &b) {
                        (Ok(x), stages) => HttpResponse::Ok()
                            .json(LULinearEquationsResponse{
                                x,
                                stages
                            }),
                        (Err(e), stages) => HttpResponse::Ok()
                            .json(LULinearEquationsFail {
                                err: e,
                                stages
                            }),
                    };
                },
                Method::Cholesky => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let n = json.n.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match cholesky(&a, &b) {
                        (Ok(x), stages) => HttpResponse::Ok()
                            .json(ComplexLULinearEquationsResponse{
                                x,
                                stages
                            }),
                        (Err(e), stages) => HttpResponse::Ok()
                            .json(ComplexLULinearEquationsFail {
                                err: e,
                                stages
                            }),
                    };
                },
                Method::Crout => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let n = json.n.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match crout(&a, &b) {
                        (Ok(x), stages) => HttpResponse::Ok()
                            .json(LULinearEquationsResponse{
                                x,
                                stages
                            }),
                        (Err(e), stages) => HttpResponse::Ok()
                            .json(LULinearEquationsFail {
                                err: e,
                                stages
                            }),
                    };
                },
                Method::Doolittle => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let n = json.n.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match doolittle(&a, &b) {
                        (Ok(x), stages) => HttpResponse::Ok()
                            .json(LULinearEquationsResponse{
                                x,
                                stages
                            }),
                        (Err(e), stages) => HttpResponse::Ok()
                            .json(LULinearEquationsFail {
                                err: e,
                                stages
                            }),
                    };
                },
                Method::GaussSeidel => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let v0_raw = json.v0.clone().unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();

                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let v0 = match Array1::from_shape_vec(n, v0_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match iterate(&a, &b, &v0, IterationType::GaussSeidel, tol, n) {
                        (Ok((x, radius)), table) => HttpResponse::Ok()
                            .json(IterativeLULinearEquationsResponse{
                                x,
                                stages: table,
                                spectral_radius: radius,
                            }),
                        (Err(e), table) => HttpResponse::Ok()
                            .json(IterativeLULinearEquationsFail {
                                err: e,
                                stages: table,
                            }),
                    };
                },
                Method::Jacobi => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let v0_raw = json.v0.clone().unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();

                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let v0 = match Array1::from_shape_vec(n, v0_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match iterate(&a, &b, &v0, IterationType::Jacobi, tol, n) {
                        (Ok((x, radius)), table) => HttpResponse::Ok()
                            .json(IterativeLULinearEquationsResponse{
                                x,
                                stages: table,
                                spectral_radius: radius,
                            }),
                        (Err(e), table) => HttpResponse::Ok()
                            .json(IterativeLULinearEquationsFail {
                                err: e,
                                stages: table,
                            }),
                    };
                },
                Method::SOR => {
                    let a_raw = json.a.clone().unwrap();
                    let b_raw = json.b.clone().unwrap();
                    let v0_raw = json.v0.clone().unwrap();
                    let n = json.n.unwrap();
                    let tol = json.tol.unwrap();
                    let w = json.w.unwrap();
                    let a = match Array2::from_shape_vec((n, n), a_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let b = match Array1::from_shape_vec(n, b_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let v0 = match Array1::from_shape_vec(n, v0_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    println!("{}\n{}", a, b);
                    return match iterate(&a, &b, &v0, IterationType::SOR(w), tol, n) {
                        (Ok((x, radius)), table) => HttpResponse::Ok()
                            .json(IterativeLULinearEquationsResponse{
                                x,
                                stages: table,
                                spectral_radius: radius,
                            }),
                        (Err(e), table) => HttpResponse::Ok()
                            .json(IterativeLULinearEquationsFail {
                                err: e,
                                stages: table,
                            }),
                    };
                },
                _ => return HttpResponse::BadRequest().body("Invalid Method"),
            }
        },
        ProblemType::Interpolation => {
            match json.method {
                Method::Vandermonde => {
                    let n = json.n.unwrap();
                    let x_raw = json.x.clone().unwrap();
                    let y_raw = json.y.clone().unwrap();
                    let x = match Array1::from_shape_vec(n, x_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let y = match Array1::from_shape_vec(n, y_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    return match vandermonde(&x, &y) {
                        Ok(pol) => HttpResponse::Ok().body(pol),
                        Err(e) => HttpResponse::Ok().json(e),
                    };

                },
                Method::DividedDifferences => {
                    let n = json.n.unwrap();
                    let x_raw = json.x.clone().unwrap();
                    let y_raw = json.y.clone().unwrap();
                    let x = match Array1::from_shape_vec(n, x_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let y = match Array1::from_shape_vec(n, y_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    return match divided_differences(&x, &y) {
                        Ok(pol) => HttpResponse::Ok().body(pol),
                        Err(e) => HttpResponse::Ok().json(e),
                    };
                },
                Method::Lagrange => {
                    let n = json.n.unwrap();
                    let x_raw = json.x.clone().unwrap();
                    let y_raw = json.y.clone().unwrap();
                    let x = match Array1::from_shape_vec(n, x_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let y = match Array1::from_shape_vec(n, y_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    return match vandermonde(&x, &y) {
                        Ok(pol) => HttpResponse::Ok().body(pol),
                        Err(e) => HttpResponse::Ok().json(e),
                    };
                },
                Method::LinearSplines => {
                    let n = json.n.unwrap();
                    let x_raw = json.x.clone().unwrap();
                    let y_raw = json.y.clone().unwrap();
                    let x = match Array1::from_shape_vec(n, x_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let y = match Array1::from_shape_vec(n, y_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    return match linear_splines(&x, &y) {
                        Ok(pol) => HttpResponse::Ok().json(pol),
                        Err(e) => HttpResponse::Ok().json(e),
                    };
                },
                Method::QuadraticSplines => {
                    let n = json.n.unwrap();
                    let x_raw = json.x.clone().unwrap();
                    let y_raw = json.y.clone().unwrap();
                    let x = match Array1::from_shape_vec(n, x_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let y = match Array1::from_shape_vec(n, y_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    return match quadratic_splines(&x, &y) {
                        Ok((splines, m)) => HttpResponse::Ok()
                            .json(SplinesMatrix {
                                splines,
                                m,
                            }),
                        Err(e) => HttpResponse::Ok().json(e),
                    };
                },
                Method::CubicSplines => {
                    let n = json.n.unwrap();
                    let x_raw = json.x.clone().unwrap();
                    let y_raw = json.y.clone().unwrap();
                    let x = match Array1::from_shape_vec(n, x_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    let y = match Array1::from_shape_vec(n, y_raw) {
                        Ok(m) => m,
                        _ => return HttpResponse::Ok().body("Bad Shape"),
                    };
                    return match cubic_splines(&x, &y) {
                        Ok((splines, m)) => HttpResponse::Ok()
                            .json(SplinesMatrix {
                                splines,
                                m,
                            }),
                        Err(e) => HttpResponse::Ok().json(e),
                    };
                },
                _ => return HttpResponse::BadRequest().body("Invalid Method"),
            }
        },
    }

    HttpResponse::Ok().body("the end, this is bad")
}

