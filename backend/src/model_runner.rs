use std::collections::HashMap;

use protobuf::Message;
use thiserror::Error;
use tract_onnx::prelude::*;
use wonnx::{
    onnx::GraphProto,
    utils::{DataTypeError, OutputTensor, Shape},
    Session,
};

use crate::ChunkSize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelChannelOrder {
    /// Batch, Channel, Height, Width order, this is the natural order for NeuraTable
    NCHW,
    /// Batch, Height, Width, Channel order, this will cause an additional permutation step
    NHWC,
}

impl ModelChannelOrder {
    fn translate_shape_to_chunksize(&self, shape: Shape) -> ChunkSize {
        match self {
            ModelChannelOrder::NCHW => ChunkSize {
                width: shape.dim(3) as usize,
                height: shape.dim(2) as usize,
            },
            ModelChannelOrder::NHWC => ChunkSize {
                width: shape.dim(2) as usize,
                height: shape.dim(1) as usize,
            },
        }
    }

    fn scratchpad_buffer_layout(&self, chunksize: ChunkSize) -> (usize, usize, usize) {
        match self {
            ModelChannelOrder::NCHW => (3, chunksize.height, chunksize.width),
            ModelChannelOrder::NHWC => (chunksize.height, chunksize.width, 3),
        }
    }

    fn get_width(&self, shape: &Shape) -> Option<usize> {
        let has_batch = shape.rank() == 4;
        shape.dims.get(self.get_width_idx(has_batch)).map(|&d| d as usize)
    }

    fn get_height(&self, shape: &Shape) -> Option<usize> {
        let has_batch = shape.rank() == 4;
        shape.dims.get(self.get_height_idx(has_batch)).map(|&d| d as usize)
    }

    fn get_batchsize(&self, shape: &Shape) -> Option<usize> {
        let has_batch = shape.rank() == 4;
        if has_batch {
            shape.dims.get(0).map(|&d| d as usize)
        } else {
            None
        }
    }

    fn get_channels(&self, shape: &Shape) -> Option<usize> {
        let has_batch = shape.rank() == 4;
        shape.dims.get(self.get_channel_idx(has_batch)).map(|&d| d as usize)
    }

    fn get_width_idx(&self, batch: bool) -> usize {
        let with_batch = match self {
            ModelChannelOrder::NCHW => 3,
            ModelChannelOrder::NHWC => 2,
        };

        if batch {
            with_batch
        } else {
            with_batch - 1
        }
    }

    fn get_height_idx(&self, batch: bool) -> usize {
        let with_batch = match self {
            ModelChannelOrder::NCHW => 2,
            ModelChannelOrder::NHWC => 1,
        };

        if batch {
            with_batch
        } else {
            with_batch - 1
        }
    }

    fn get_channel_idx(&self, batch: bool) -> usize {
        let with_batch = match self {
            ModelChannelOrder::NCHW => 1,
            ModelChannelOrder::NHWC => 3,
        };

        if batch {
            with_batch
        } else {
            with_batch - 1
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelRunnerError {
    #[error("The model has too many inputs")]
    ModelInputError,
    #[error("The models input is unsupported. A [1,3,h,w] or [1,h,w,3] shaped input is required (NCHW or NHWC).")]
    InvalidInputShape(Shape),
    #[error("Could not read model parameters")]
    ModelParameterError(#[from] DataTypeError),
    #[error("The model has no suitable output")]
    NoSuitableOutput,
    #[error("The model is not parseable")]
    ParseError(#[from] protobuf::ProtobufError),
}

pub struct WonnxRunner {
    session: Session,
    input_name: String,
    output_name: String,
    input_scratchpad: ndarray::Array3<f32>,
}

pub struct TractRunner {
    model: Box<dyn Fn(&ndarray::Array3<f32>, &[usize]) -> ndarray::Array3<f32>>,
    input_scratchpad: ndarray::Array3<f32>,
}

pub enum ModelRunnerBackend {
    WonnxRunner(WonnxRunner),
    TractRunner(TractRunner),
}

pub struct ModelRunner {
    backend: ModelRunnerBackend,
    chunksize: ChunkSize,
    model_channel_order: ModelChannelOrder,
    model_scale: usize,
}

impl ModelRunner {
    pub fn get_chunksize(&self) -> ChunkSize {
        self.chunksize
    }

    fn get_graph_input(
        graph: &GraphProto,
    ) -> Result<(Shape, String, ModelChannelOrder), ModelRunnerError> {
        let inputs = graph.get_input();

        if inputs.len() > 1 {
            return Err(ModelRunnerError::ModelInputError);
        }
        let input_shape = inputs[0].get_shape()?;
        let input_name = inputs[0].get_name().to_owned();

        if input_shape.dim(0) != 1 {
            return Err(ModelRunnerError::InvalidInputShape(input_shape));
        }

        let channel_order = if input_shape.dim(1) == 3 {
            log::debug!("NCHW model detected!");
            ModelChannelOrder::NCHW
        } else if input_shape.dim(3) == 3 {
            log::debug!("NHWC model detected!");
            ModelChannelOrder::NHWC
        } else {
            return Err(ModelRunnerError::InvalidInputShape(input_shape));
        };

        Ok((input_shape, input_name, channel_order))
    }

    fn get_scale_factor(
        input_shape: &Shape,
        model_channel_order: ModelChannelOrder,
        output_shape: &Shape,
    ) -> Option<usize> {
        if model_channel_order.get_batchsize(input_shape)
            == model_channel_order.get_batchsize(output_shape)
            && model_channel_order.get_channels(input_shape)
                == model_channel_order.get_channels(output_shape)
        {
            let in_width = model_channel_order.get_width(input_shape)?;
            let out_width = model_channel_order.get_width(output_shape)?;
            let in_height = model_channel_order.get_height(input_shape)?;
            let out_height = model_channel_order.get_height(output_shape)?;

            let xscale = out_width / in_width;
            let yscale = out_height / in_height;
            if xscale == yscale
                && in_width * xscale == out_width
                && in_height * yscale == out_height
            {
                Some(xscale)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_matching_output(
        graph: &GraphProto,
        input_shape: &Shape,
        channel_order: ModelChannelOrder,
    ) -> Result<(String, usize), ModelRunnerError> {
        let exact_match = graph
            .get_output()
            .iter()
            .filter(|o| o.get_shape().map(|s| &s == input_shape).unwrap_or_default())
            .next()
            .map(|o| (o.get_name().to_owned(), 1));

        exact_match
            .or_else(|| {
                graph
                    .get_output()
                    .iter()
                    .find_map(|output| match output.get_shape() {
                        Ok(output_shape) => {
                            Self::get_scale_factor(input_shape, channel_order, &output_shape)
                                .map(|scale| (output.get_name().to_owned(), scale))
                        }
                        Err(_) => None,
                    })
            })
            .ok_or_else(|| ModelRunnerError::NoSuitableOutput)
    }

    pub async fn new<R>(input: &mut R, force_tract: bool) -> Result<Self, ModelRunnerError>
    where
        R: std::io::Read + std::io::Seek,
    {
        let wonnx_model = wonnx::onnx::ModelProto::parse_from_reader(input)?;

        let graph = wonnx_model.get_graph();
        let (input_shape, input_name, model_channel_order) = Self::get_graph_input(graph)?;
        log::info!("Detected model input shape: {:?}", input_shape);
        let (output_name, model_scale) =
            Self::get_matching_output(graph, &input_shape, model_channel_order)?;
        log::info!(
            "Using output {} with {}x scaling",
            &output_name,
            model_scale
        );
        let chunksize = model_channel_order.translate_shape_to_chunksize(input_shape);

        if !force_tract {
            match Session::from_model(wonnx_model).await {
                Ok(session) => {
                    return Ok(Self {
                        backend: ModelRunnerBackend::WonnxRunner(WonnxRunner {
                            session,
                            input_name,
                            output_name,
                            input_scratchpad: ndarray::Array3::<f32>::zeros(
                                model_channel_order.scratchpad_buffer_layout(chunksize),
                            ),
                        }),
                        chunksize,
                        model_channel_order,
                        model_scale,
                    })
                }
                Err(err) => {
                    log::error!("Failed to create wonnx session: {}", err);
                    log::error!("Either wonnx doesn't support your model right now or you don't have Vulkan available. We will fall back to tract, but this will be slow!");
                }
            }
        }
        input.rewind().unwrap();

        let tract_model = tract_onnx::onnx()
            .model_for_read(input)
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();

        let infer = move |input: &ndarray::Array3<f32>, output_shape: &[usize]| {
            let shape = input.shape().clone();
            let mut result = tract_model
                .run(tvec![Into::<Tensor>::into(
                    input
                        .clone()
                        .into_shape((1, shape[0], shape[1], shape[2]))
                        .unwrap()
                )
                .into()])
                .unwrap();
            result
                .remove(0)
                .into_tensor()
                .into_array()
                .unwrap()
                .into_shape((output_shape[0], output_shape[1], output_shape[2]))
                .unwrap()
        };

        Ok(Self {
            backend: ModelRunnerBackend::TractRunner(TractRunner {
                model: Box::new(infer),
                input_scratchpad: ndarray::Array3::<f32>::zeros(
                    model_channel_order.scratchpad_buffer_layout(chunksize),
                ),
            }),
            chunksize,
            model_channel_order,
            model_scale,
        })
    }

    /// Scale down a chunk of image data by the given scale factor in the x and y dimension
    ///
    /// The image chunk should be in CHW channel order.
    /// The downscaling is done via simple averaging, so this should be considered a temporary
    /// solution!
    fn scale_chunk(mut chunk: ndarray::Array3<f32>, scale: usize) -> ndarray::Array3<f32> {
        let shape: Vec<_> = chunk.shape().iter().cloned().collect();

        for c in 0..shape[0] {
            for y in 0..shape[1] / scale {
                for x in 0..shape[2] / scale {
                    let mut nv = 0f32;
                    for dy in 0..scale {
                        for dx in 0..scale {
                            nv += chunk[(c, (scale * y) + dy, (scale * x) + dx)];
                        }
                    }
                    chunk[(c, y, x)] = nv / (scale * scale) as f32;
                }
            }
        }

        chunk.slice_move(ndarray::s![.., ..shape[1] / scale, ..shape[2] / scale])
    }

    pub async fn process_chunk<'a>(
        &mut self,
        input: ndarray::ArrayView3<'a, f32>,
    ) -> Result<ndarray::Array3<f32>, ModelRunnerError> {

        // Input will be an ArrayView to an array of shape (CHW)
        let model_order_input = match self.model_channel_order {
            ModelChannelOrder::NCHW => input,
            ModelChannelOrder::NHWC => input.permuted_axes([1, 2, 0]),
        };

        let mut model_output_shape: Vec<_> = model_order_input.shape().iter().cloned().collect();
        model_output_shape[self.model_channel_order.get_width_idx(false)] *= self.model_scale;
        model_output_shape[self.model_channel_order.get_height_idx(false)] *= self.model_scale;

        let model_output = match &mut self.backend {
            ModelRunnerBackend::WonnxRunner(runner) => {
                runner
                    .process_chunk(model_order_input, model_output_shape.as_slice())
                    .await?
            }
            ModelRunnerBackend::TractRunner(runner) => {
                runner
                    .process_chunk(model_order_input, model_output_shape.as_slice())
                    .await?
            }
        };

        let mut nchw_output = match self.model_channel_order {
            ModelChannelOrder::NCHW => model_output,
            ModelChannelOrder::NHWC => model_output.permuted_axes([2, 0, 1]),
        };

        if self.model_scale > 1 {
            nchw_output = Self::scale_chunk(nchw_output, self.model_scale)
        }

        Ok(nchw_output)
    }
}

impl WonnxRunner {
    fn get_output_tensor(
        &self,
        network_result: &mut HashMap<String, OutputTensor>,
        shape: &[usize],
    ) -> ndarray::Array3<f32> {
        if let OutputTensor::F32(data) = network_result.remove(&self.output_name).unwrap() {
            ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), data).unwrap()
        } else {
            panic!("Unexpected output type!");
        }
    }

    pub async fn process_chunk<'a>(
        &mut self,
        input: ndarray::ArrayView3<'a, f32>,
        output_shape: &[usize],
    ) -> Result<ndarray::Array3<f32>, ModelRunnerError> {
        input.assign_to(&mut self.input_scratchpad);
        let input_map = HashMap::from([(
            self.input_name.clone(),
            self.input_scratchpad.as_slice().unwrap().into(),
        )]);
        let mut result = self.session.run(&input_map).await.unwrap();

        Ok(self.get_output_tensor(&mut result, output_shape))
    }
}

impl TractRunner {
    pub async fn process_chunk<'a>(
        &mut self,
        input: ndarray::ArrayView3<'a, f32>,
        output_shape: &[usize],
    ) -> Result<ndarray::Array3<f32>, ModelRunnerError> {
        input.assign_to(&mut self.input_scratchpad);
        Ok((self.model)(&self.input_scratchpad, output_shape))
    }
}
