#[derive(Debug, Clone, Copy)]
pub struct ChunkSize {
    pub width: usize,
    pub height: usize,
}

impl ChunkSize {
    pub fn as_pair(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn remaining_area_after_padding(&self, padding: usize) -> Self {
        Self {
            height: self.height - 2 * padding,
            width: self.width - 2 * padding,
        }
    }

    pub fn stepsize_with_overlap(&self, overlap: usize) -> Self {
        Self {
            height: self.height - overlap,
            width: self.width - overlap,
        }
    }
}
