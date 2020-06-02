extern crate abaq_methods;
use actix_web::{post, get, web, App, HttpResponse, HttpServer, Responder};
use serde::{Serialize, Deserialize};

#[derive(Deserialize)]
enum AbaQRequest {

}

#[derive(Serialize, Deserialize)]
enum RootFindingErr {
    Absolute,
    Relative,
}

#[derive(Deserialize)]
enum RootFindingReq {
    IncrementalSearch {
        f: String,
        x0: f64,
        dx: f64,
        n: usize,
    },
    Bisection {
        f: String,
        xl: f64,
        xu: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
    FalsePosition {
        f: String,
        xl: f64,
        xu: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
    FixedPoint {
        f: String,
        g: String,
        x0: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
    Newton {
        f: String,
        df: String,
        x0: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
    Secant {
        f: String,
        x0: f64,
        x1: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
    MultipleRoot {
        f: String,
        df: String,
        d2f: String,
        x0: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
    Steffensen {
        f: String,
        x: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
    Muller {
        f: String,
        x0: f64,
        x1: f64,
        x2: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
    AcceleratedFixedPoint {
        f: String,
        g: String,
        x0: f64,
        tol: f64,
        n: usize,
        err: RootFindingErr,
    },
}

#[derive(Serialize, Deserialize)]
struct RequestTest {
    f: Option<String>,
    g: Option<String>,
    df: Option<String>,
    d2f: Option<String>,
    x0: Option<f64>,
    x1: Option<f64>,
    x2: Option<f64>,
    tol: Option<f64>,
    n: Option<usize>,
    err: Option<RootFindingErr>,
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
    format!("Hello biches")
}

#[post("/api")]
async fn api(json: web::Json<RequestTest>,) -> impl Responder {
    json
}
