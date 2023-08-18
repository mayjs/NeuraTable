use std::collections::HashMap;

use super::image_chunk_iterator::ImageChunkGeneratorBuilder;
use super::model_runner::ModelRunner;
use image::{ImageBuffer, Rgb};
use ndarray::Array3;
use thiserror::Error;
use wonnx::{
    onnx::{self, GraphProto},
    utils::{DataTypeError, OutputTensor, Shape},
    Session,
};

#[derive(Debug, Error)]
pub enum ImageProcessingError {
    #[error("Session could not be created")]
    SessionCreationError(#[from] wonnx::SessionError),
    #[error("The model has too many inputs")]
    ModelInputError,
    #[error("Could not model parameters")]
    ModelParameterError(#[from] DataTypeError),
    #[error("The model has no suitable output")]
    NoSuitableOutput,
    #[error("The models input is unsupported (a [1,3,h,w] shaped input is required and h must be equal to w)")]
    InvalidInputShape(Shape),
    #[error("The chunk generator failed")]
    ChunkGeneratorError(#[from] super::image_chunk_iterator::ImageChunkGeneratorError),
}

pub struct ImageProcessor {
    runner: ModelRunner,
    chunk_size: usize,
    chunk_padding: usize,
    chunk_overlap: usize,
}

impl ImageProcessor {
    pub async fn new(runner: ModelRunner) -> Result<ImageProcessor, ImageProcessingError> {
        let chunk_size = runner.get_chunksize();

        let default_padding = chunk_size / 7; // TODO: This is an experimental value and will probably to
                                              // work for many models
        let default_overlap = default_padding / 10;

        Ok(ImageProcessor {
            runner,
            chunk_size,
            chunk_padding: default_padding,
            chunk_overlap: default_overlap,
        })
    }

    pub async fn process_image(
        &mut self,
        image: ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, ImageProcessingError> {
        let width = image.width() as usize;
        let height = image.height() as usize;

        let image_data = Array3::from_shape_vec((height, width, 3), image.into_raw())
            .unwrap()
            .mapv(|v| (v as f32) / 255.0)
            .permuted_axes([2, 0, 1]); // The image data comes in HxWxC format, we need CxHxW

        let generator = ImageChunkGeneratorBuilder::new_from_array(image_data)
            .with_chunksize(self.chunk_size)
            .with_chunk_padding(self.chunk_padding)
            .with_overlap(self.chunk_overlap)
            .finalize()?;

        // Caution: We create the output buffer in the image layout directly, that way we won't
        // have to worry about permutation when creating the resulting image
        let mut output_image: Array3<f32> = Array3::zeros((height, width, 3));

        for (i, chunk) in generator.iter().enumerate() {
            log::info!("Processing chunk {}", i);

            let mut result_tensor = self.runner.process_chunk(chunk.chunk).await.unwrap();

            let mut usable_output_chunk = result_tensor.slice_mut(chunk.get_usable_range());
            generator.scale_overlap(&chunk.global_coordinate_offset, &mut usable_output_chunk);
            let mut output_range = output_image.slice_mut(ndarray::s![
                chunk.global_coordinate_offset.y
                    ..chunk.global_coordinate_offset.y + usable_output_chunk.shape()[1],
                chunk.global_coordinate_offset.x
                    ..chunk.global_coordinate_offset.x + usable_output_chunk.shape()[2],
                ..,
            ]);
            // Since the network returns data in CxHxW order, we need to permute to HxWxC order
            output_range += &usable_output_chunk.permuted_axes([1, 2, 0]);
        }

        let raw_output_image_data = output_image.mapv(|v| (v * 255.0) as u8);
        Ok(ImageBuffer::from_raw(
            width as u32,
            height as u32,
            raw_output_image_data.into_raw_vec(),
        )
        .unwrap())
    }
}

