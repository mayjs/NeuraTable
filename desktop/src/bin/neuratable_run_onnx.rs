use std::str::FromStr;

use argh::FromArgs;
use backend::image_processor::{ImageColorModel, ImageProcessor, ModelValueRange};

#[derive(Debug, Clone, PartialEq)]
struct ArgColorModel(ImageColorModel);

impl FromStr for ArgColorModel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let uppercase = s.to_uppercase();
        Ok(match uppercase.as_ref() {
            "BGR" => ArgColorModel(ImageColorModel::BGR),
            "RGB" => ArgColorModel(ImageColorModel::RGB),
            _ => anyhow::bail!("Color model {} not known, must be one of (RGB, BGR)", s),
        })
    }
}

#[derive(FromArgs, PartialEq, Debug)]
/// Run a 1:1 ONNX model in chunked mode
struct RunOnnx {
    #[argh(positional)]
    onnx_model: String,
    #[argh(positional)]
    input_image: String,
    #[argh(positional)]
    output_image: String,
    /// the expected color channel order of the model
    #[argh(option, default = "ArgColorModel(ImageColorModel::RGB)")]
    model_channel_order: ArgColorModel,
    /// whether or not to force CPU processing
    #[argh(switch)]
    force_cpu: bool,
}

async fn run(args: RunOnnx) {
    let mut r = std::fs::File::open(&args.onnx_model).unwrap();

    let runner = backend::model_runner::ModelRunner::new(&mut r, args.force_cpu)
        .await
        .unwrap();

    // FIXME: These must be parsed from arguments
    let input_range = ModelValueRange::asymmetric(255.0);
    let output_range = ModelValueRange::asymmetric(255.0);

    let mut processor = ImageProcessor::new(
        runner,
        args.model_channel_order.0,
        input_range,
        output_range,
    )
    .await
    .unwrap();

    let input_image = image::open(&args.input_image).unwrap().to_rgb16();
    let output_image = processor.process_image(input_image).await.unwrap();

    // FIXME: For JPG Output, we need to scale the image data back to 8 Bit RGB
    // We need to find a generic way to solve this issue
    output_image.save(&args.output_image).unwrap();
}

fn main() {
    env_logger::init();
    log::debug!("Test");
    let args: RunOnnx = argh::from_env();
    pollster::block_on(run(args));
}
