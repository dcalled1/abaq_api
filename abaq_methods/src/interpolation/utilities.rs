
#[derive(Debug)]
pub enum Error {
    BadIn,
}

#[derive(Debug)]
pub struct Spline {
    start: f64,
    end: f64,
    polynomial: String,
}

#[derive(Debug)]
pub struct Splines {
    splines: Vec<Spline>,
}

impl Splines {
    pub fn new() -> Splines {
        Splines {
            splines: Vec::<Spline>::new(),
        }
    }

    pub fn add(&mut self, start: f64, end: f64, pol: &String) {
        self.splines.push(Spline {
            start,
            end,
            polynomial: pol.clone(),
        })
    }
}