use argh::FromArgs;
use backend::image_processor::ImageProcessor;

#[derive(FromArgs, PartialEq, Debug)]
/// Run a 1:1 ONNX model in chunked mode
struct RunOnnx {
    #[argh(positional)]
    onnx_model: String,
    #[argh(positional)]
    input_image: String,
    #[argh(positional)]
    output_image: String,
    /// whether or not to force CPU processing
    #[argh(switch)]
    force_cpu: bool,
}

async fn run(args: RunOnnx) {
    let mut r = std::fs::File::open(&args.onnx_model).unwrap();

    let runner = backend::model_runner::ModelRunner::new(&mut r, args.force_cpu)
        .await
        .unwrap();

    let mut processor = ImageProcessor::new(runner).await.unwrap();

    let input_image = image::open(&args.input_image).unwrap().to_rgb8();
    let output_image = processor.process_image(input_image).await.unwrap();

    output_image.save(&args.output_image).unwrap();
}

fn main() {
    env_logger::init();
    log::debug!("Test");
    let args: RunOnnx = argh::from_env();
    pollster::block_on(run(args));
}
