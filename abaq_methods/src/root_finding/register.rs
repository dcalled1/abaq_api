use std::fmt;

pub struct Log {
    vars: Vec<f64>,
    i: u32
}
pub struct Logbook {
    head: Vec<String>,
    regs: Vec<Log>,
}

impl Log {
    fn new(i: u32, vars: Vec<f64>) -> Log {
        Log{i, vars}
    }
}

impl Logbook {
    pub(crate) fn new(n: u32, head: Vec<String>) -> Logbook {
        Logbook{regs: Vec::with_capacity(n as usize), head}
    }

    pub(crate) fn registry(&mut self, i: u32, vars: Vec<f64>) {
        self.regs.push(Log::new(i, vars));
    }
}

impl fmt::Display for Log {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        s.push_str(format!(" {:^5} ", self.i).as_str());
        for v in &self.vars {
            s.push_str(format!(" {:^30} ", v).as_str());
        }
        s.push('\n');
        write!(f, "{}", s)
    }
}

impl fmt::Display for Logbook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        s.push_str(format!(" {:^5} ", "i").as_str());
        for v in &self.head {
            s.push_str(format!(" {:^30} ", v).as_str());
        }
        s.push('\n');
        for v in &self.regs {
            s.push_str(format!(" {:^15} ", v).as_str());
        }
        write!(f, "{}", s)
    }
}