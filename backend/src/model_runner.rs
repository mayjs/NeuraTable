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
    model: Box<dyn Fn(&ndarray::Array3<f32>) -> ndarray::Array3<f32>>,
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

    fn get_matching_output(
        graph: &GraphProto,
        input_shape: &Shape,
    ) -> Result<String, ModelRunnerError> {
        graph
            .get_output()
            .iter()
            .filter(|o| o.get_shape().map(|s| &s == input_shape).unwrap_or_default())
            .next()
            .map(|o| o.get_name().to_owned())
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
        let output_name = Self::get_matching_output(graph, &input_shape)?;
        log::info!("Using output {}", &output_name);
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

        let infer = move |input: &ndarray::Array3<f32>| {
            let shape = input.shape().clone();
            log::debug!("Running tract model with input shape {:?}", shape);
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
                .into_shape((shape[0], shape[1], shape[2]))
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
        })
    }

    pub async fn process_chunk<'a>(
        &mut self,
        input: ndarray::ArrayView3<'a, f32>,
    ) -> Result<ndarray::Array3<f32>, ModelRunnerError> {
        log::debug!("Application order data shape: {:?}", input.shape());

        // Input will be an ArrayView to an array of shape (CHW)
        let model_order_input = match self.model_channel_order {
            ModelChannelOrder::NCHW => input,
            ModelChannelOrder::NHWC => input.permuted_axes([1, 2, 0]),
        };

        log::debug!("Model order data shape: {:?}", model_order_input.shape());

        let model_output = match &mut self.backend {
            ModelRunnerBackend::WonnxRunner(runner) => {
                runner.process_chunk(model_order_input).await?
            }
            ModelRunnerBackend::TractRunner(runner) => {
                runner.process_chunk(model_order_input).await?
            }
        };

        Ok(match self.model_channel_order {
            ModelChannelOrder::NCHW => model_output,
            ModelChannelOrder::NHWC => model_output.permuted_axes([2, 0, 1]),
        })
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
    ) -> Result<ndarray::Array3<f32>, ModelRunnerError> {
        input.assign_to(&mut self.input_scratchpad);
        let input_map = HashMap::from([(
            self.input_name.clone(),
            self.input_scratchpad.as_slice().unwrap().into(),
        )]);
        let mut result = self.session.run(&input_map).await.unwrap();
        Ok(self.get_output_tensor(&mut result, input.shape()))
    }
}

impl TractRunner {
    pub async fn process_chunk<'a>(
        &mut self,
        input: ndarray::ArrayView3<'a, f32>,
    ) -> Result<ndarray::Array3<f32>, ModelRunnerError> {
        input.assign_to(&mut self.input_scratchpad);
        Ok((self.model)(&self.input_scratchpad))
    }
}