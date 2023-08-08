use std::collections::HashMap;

use super::image_chunk_iterator::ImageChunkGeneratorBuilder;
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
    wonnx_session: Session,
    chunk_size: usize,
    input_name: String,
    output_name: String,
    chunk_padding: usize,
    chunk_overlap: usize,
}

impl ImageProcessor {
    fn get_graph_input(graph: &GraphProto) -> Result<(Shape, String), ImageProcessingError> {
        let inputs = graph.get_input();

        if inputs.len() > 1 {
            return Err(ImageProcessingError::ModelInputError);
        }
        let input_shape = inputs[0].get_shape()?;
        let input_name = inputs[0].get_name().to_owned();

        if input_shape.dim(0) != 1
            || input_shape.dim(1) != 3
            || input_shape.dim(2) != input_shape.dim(3)
        {
            return Err(ImageProcessingError::InvalidInputShape(input_shape));
        }

        Ok((input_shape, input_name))
    }

    fn get_matching_output(
        graph: &GraphProto,
        input_shape: &Shape,
    ) -> Result<String, ImageProcessingError> {
        graph
            .get_output()
            .iter()
            .filter(|o| o.get_shape().map(|s| &s == input_shape).unwrap_or_default())
            .next()
            .map(|o| o.get_name().to_owned())
            .ok_or_else(|| ImageProcessingError::NoSuitableOutput)
    }

    pub async fn new(model: onnx::ModelProto) -> Result<ImageProcessor, ImageProcessingError> {
        let graph = model.get_graph();

        let (input_shape, input_name) = Self::get_graph_input(graph)?;
        let output_name = Self::get_matching_output(graph, &input_shape)?;
        log::info!("Using output {}", &output_name);

        let chunk_size = input_shape.dim(3) as usize;

        let wonnx_session = Session::from_model(model).await?;

        let default_padding = chunk_size / 7; // TODO: This is an experimental value and will probably to
                                              // work for many models
        let default_overlap = default_padding / 10;

        Ok(ImageProcessor {
            wonnx_session,
            chunk_size,
            input_name,
            output_name,
            chunk_padding: default_padding,
            chunk_overlap: default_overlap,
        })
    }

    fn get_output_tensor(&self, network_result: &mut HashMap<String, OutputTensor>) -> Array3<f32> {
        if let OutputTensor::F32(data) = network_result.remove(&self.output_name).unwrap() {
            Array3::from_shape_vec((3, self.chunk_size, self.chunk_size), data).unwrap()
        } else {
            panic!("Unexpected output type!");
        }
    }

    pub async fn process_image(
        &self,
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

        let mut input_data: Array3<f32> = Array3::zeros((3, self.chunk_size, self.chunk_size));
        for (i, chunk) in generator.iter().enumerate() {
            log::info!("Processing chunk {}", i);
            chunk.chunk.assign_to(&mut input_data);
            let input_map = HashMap::from([(
                self.input_name.clone(),
                input_data.as_slice().unwrap().into(),
            )]);

            let mut result = self.wonnx_session.run(&input_map).await?;
            let mut result_tensor = self.get_output_tensor(&mut result);

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

