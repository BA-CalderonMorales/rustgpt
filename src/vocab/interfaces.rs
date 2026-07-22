use std::collections::HashMap;

use bincode::Encode;

#[derive(Clone, Encode)]
pub struct Vocab {
    pub encode: HashMap<String, usize>,
    pub decode: HashMap<usize, String>,
    pub words: Vec<String>,
}
