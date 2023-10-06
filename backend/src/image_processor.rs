use crate::{model_value_range::ModelValueRange, ChunkSize};

use super::image_chunk_iterator::ImageChunkGeneratorBuilder;
use super::model_runner::ModelRunner;
use image::{ImageBuffer, Rgb};
use ndarray::Array3;
use thiserror::Error;
use wonnx::utils::{DataTypeError, Shape};

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
    model_color_model: ImageColorModel,
    model_input_range: ModelValueRange,
    model_output_range: ModelValueRange,
    chunksize: ChunkSize,
    chunk_padding: usize,
    chunk_overlap: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageColorModel {
    RGB,
    BGR,
}

impl ImageProcessor {
    pub async fn new(
        runner: ModelRunner,
        model_color_model: ImageColorModel,
        model_input_range: ModelValueRange,
        model_output_range: ModelValueRange,
    ) -> Result<ImageProcessor, ImageProcessingError> {
        let chunksize = runner.get_chunksize();

        let min_dim = std::cmp::min(chunksize.width, chunksize.height);

        let default_padding = min_dim / 7; // TODO: This is an experimental value and will probably to
                                           // work for many models
        let default_overlap = default_padding / 10;

        Ok(ImageProcessor {
            runner,
            model_color_model,
            model_input_range,
            model_output_range,
            chunksize,
            chunk_padding: default_padding,
            chunk_overlap: default_overlap,
        })
    }

    /// Change the color channel order of an image in RGB to BGR (or vice versa)
    ///
    /// The data channel order must be in HxWxC order (i.e. height x width x 3)
    ///
    /// This is very inefficient, but to make it more eficient would probably take unsafe code.
    /// Maybe we could look into adding a "permute_axis" function to ndarray.
    fn rgb_to_bgr<T>(data: &mut Array3<T>) {
        log::debug!(
            "Swapping the first and third index of the third axis in data shape {:?}",
            data.shape()
        );
        for y in 0..data.shape()[0] {
            for x in 0..data.shape()[1] {
                data.swap((y, x, 0), (y, x, 2))
            }
        }
    }

    pub async fn process_image(
        &mut self,
        image: ImageBuffer<Rgb<u16>, Vec<u16>>,
    ) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>, ImageProcessingError> {
        let width = image.width() as usize;
        let height = image.height() as usize;

        let mut image_data = Array3::from_shape_vec((height, width, 3), image.into_raw())
            .unwrap()
            .mapv(|v| self.model_input_range.pixel_value_to_model(v));
        if self.model_color_model == ImageColorModel::BGR {
            Self::rgb_to_bgr(&mut image_data);
        }
        image_data = image_data.permuted_axes([2, 0, 1]); // The image data comes in HxWxC format, we need CxHxW

        let generator = ImageChunkGeneratorBuilder::new_from_array(image_data)
            .with_chunksize(self.chunksize)
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

        log::debug!("Output Mean: {}", output_image.mean().unwrap());
        self.model_output_range
            .normalize_model_value(&mut output_image);

        let mut raw_output_image_data = output_image.mapv(|v| (v * u16::MAX as f32) as u16);
        if self.model_color_model == ImageColorModel::BGR {
            Self::rgb_to_bgr(&mut raw_output_image_data);
        }
        Ok(ImageBuffer::from_raw(
            width as u32,
            height as u32,
            raw_output_image_data.into_raw_vec(),
        )
        .unwrap())
    }
}
