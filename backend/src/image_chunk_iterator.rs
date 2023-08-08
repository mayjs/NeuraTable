use std::{cmp::min, marker::PhantomData};

use ndarray::{s, Array3, ArrayView3, ArrayViewMut3, Dim, Ix3, SliceArg};
use ndarray_ndimage::PadMode;
use thiserror::Error;

pub struct Finalized;

pub type ImageTensor = Array3<f32>;

pub struct ImageChunkGenerator<M> {
    image_data: ImageTensor,
    chunksize: usize,
    overlap: usize,
    chunk_padding: usize,
    input_image_resolution: (usize, usize),
    input_image_padding: (usize, usize),
    _marker: PhantomData<M>,
}

pub type ImageChunkGeneratorBuilder = ImageChunkGenerator<()>;
pub type FinalizedImageChunkGenerator = ImageChunkGenerator<Finalized>;

pub struct ImageChunkIterator<'a> {
    data: &'a FinalizedImageChunkGenerator,
    current_coords: (usize, usize),
}

pub struct Coords {
    pub x: usize,
    pub y: usize,
}

pub struct ImageChunk<'a> {
    pub chunk: ArrayView3<'a, f32>,
    pub global_coordinate_offset: Coords,
    pub gen: &'a FinalizedImageChunkGenerator,
}

impl<'a> Iterator for ImageChunkIterator<'a> {
    type Item = ImageChunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let useful_chunksize = self.data.chunksize - 2 * self.data.chunk_padding;
        let step_size = useful_chunksize - self.data.overlap;

        if self.current_coords.1 < self.data.input_image_resolution.1 {
            let mut x = self.current_coords.0 + self.data.input_image_padding.0;
            let mut y = self.current_coords.1 + self.data.input_image_padding.1;

            x -= self.data.chunk_padding;
            y -= self.data.chunk_padding;

            let chunk = self.data.image_data.slice(s![
                ..,
                y..y + self.data.chunksize,
                x..x + self.data.chunksize
            ]);
            let global_coordinate_offset = Coords {
                x: self.current_coords.0,
                y: self.current_coords.1,
            };

            self.current_coords.0 += step_size;
            if self.current_coords.0 >= self.data.input_image_resolution.0 {
                self.current_coords.0 = 0;
                self.current_coords.1 += step_size;
            }

            Some(ImageChunk {
                chunk,
                global_coordinate_offset,
                gen: &self.data,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum ImageChunkGeneratorError {
    #[error("Padding {0} exceeds chunksize {1}")]
    InvalidPaddingValue(usize, usize),
    #[error("Overlap {0} exceeds usable chunk area {1}")]
    InvalidOverlapValue(usize, usize),
}

impl ImageChunkGeneratorBuilder {
    pub fn new_from_array(image: ImageTensor) -> Self {
        Self {
            image_data: image,
            chunksize: 440, // Default values from nind-denoise
            overlap: 6,
            chunk_padding: 60,
            input_image_resolution: (0, 0), // We will calculate the actual size of these when
            // finalizing
            input_image_padding: (0, 0),
            _marker: PhantomData,
        }
    }

    pub fn set_chunksize(&mut self, chunksize: usize) {
        self.chunksize = chunksize;
    }

    pub fn with_chunksize(mut self, chunksize: usize) -> Self {
        self.set_chunksize(chunksize);
        self
    }

    pub fn set_chunk_padding(&mut self, chunk_padding: usize) {
        self.chunk_padding = chunk_padding;
    }

    pub fn with_chunk_padding(mut self, chunk_padding: usize) -> Self {
        self.set_chunk_padding(chunk_padding);
        self
    }

    pub fn set_overlap(&mut self, overlap: usize) {
        self.overlap = overlap;
    }

    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.set_overlap(overlap);
        self
    }

    fn pad_image(&mut self) {
        let needed_padding = self.chunksize;
        self.image_data = ndarray_ndimage::pad(
            &self.image_data,
            &[
                [0, 0],
                [needed_padding, needed_padding],
                [needed_padding, needed_padding],
            ],
            PadMode::Reflect,
        );
        self.input_image_padding = (needed_padding, needed_padding);
    }

    pub fn finalize(mut self) -> Result<FinalizedImageChunkGenerator, ImageChunkGeneratorError> {
        if 2 * self.chunk_padding >= self.chunksize {
            return Err(ImageChunkGeneratorError::InvalidPaddingValue(
                self.chunk_padding,
                self.chunksize,
            ));
        }

        let usable_output_chunksize = self.chunksize - 2 * self.chunk_padding;
        if 2 * self.overlap > usable_output_chunksize {
            return Err(ImageChunkGeneratorError::InvalidOverlapValue(
                self.overlap,
                usable_output_chunksize,
            ));
        }

        self.input_image_resolution = (self.image_data.shape()[2], self.image_data.shape()[1]);
        self.pad_image();

        Ok(FinalizedImageChunkGenerator {
            image_data: self.image_data,
            chunksize: self.chunksize,
            overlap: self.overlap,
            chunk_padding: self.chunk_padding,
            input_image_resolution: self.input_image_resolution,
            input_image_padding: self.input_image_padding,
            _marker: PhantomData,
        })
    }
}

impl FinalizedImageChunkGenerator {
    /// Returns the useful area of image data in each chunk.
    /// The result is a pair of the inclusive range start and the exclusive range end
    pub fn useful_chunk_area(&self) -> (Coords, Coords) {
        (
            Coords {
                x: self.chunk_padding,
                y: self.chunk_padding,
            },
            Coords {
                x: self.chunksize - self.chunk_padding,
                y: self.chunksize - self.chunk_padding,
            },
        )
    }

    pub fn iter(&self) -> ImageChunkIterator {
        ImageChunkIterator {
            data: self,
            current_coords: (0, 0),
        }
    }

    pub fn scale_overlap(&self, global_coords: &Coords, chunk: &mut ArrayViewMut3<'_, f32>) {
        if global_coords.x > 0 {
            *(&mut chunk.slice_mut(s![.., .., 0..self.overlap])) *= 0.5;
        }
        if global_coords.y > 0 {
            *(&mut chunk.slice_mut(s![.., 0..self.overlap, ..])) *= 0.5;
        }
        if global_coords.x + self.chunksize - 2 * self.chunk_padding < self.input_image_resolution.0
        {
            let start = chunk.shape()[2] - self.overlap;
            *(&mut chunk.slice_mut(s![.., .., start..start + self.overlap])) *= 0.5;
        }
        if global_coords.y + self.chunksize - 2 * self.chunk_padding < self.input_image_resolution.1
        {
            let start = chunk.shape()[1] - self.overlap;
            *(&mut chunk.slice_mut(s![.., start..start + self.overlap, ..])) *= 0.5;
        }
    }
}

impl<'a> ImageChunk<'a> {
    pub fn get_usable_range(&self) -> impl SliceArg<Ix3, OutDim = Dim<[usize; 3]>> {
        let width = min(
            self.gen.chunksize - 2 * self.gen.chunk_padding,
            self.gen.input_image_resolution.0 - self.global_coordinate_offset.x,
        );
        let height = min(
            self.gen.chunksize - 2 * self.gen.chunk_padding,
            self.gen.input_image_resolution.1 - self.global_coordinate_offset.y,
        );

        s![
            ..,
            self.gen.chunk_padding..self.gen.chunk_padding + height,
            self.gen.chunk_padding..self.gen.chunk_padding + width
        ]
    }
}
