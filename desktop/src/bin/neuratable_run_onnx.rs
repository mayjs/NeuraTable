use backend::image_processor::ImageProcessor;
use protobuf::Message;
use argh::FromArgs;

#[derive(FromArgs, PartialEq, Debug)]
/// Run a 1:1 ONNX model in chunked mode
struct RunOnnx {
    #[argh(positional)]
    onnx_model: String,
    #[argh(positional)]
    input_image: String,
    #[argh(positional)]
    output_image: String,
}

async fn run(args: RunOnnx) {
    let model = wonnx::onnx::ModelProto::parse_from_bytes(&std::fs::read(&args.onnx_model).unwrap()).unwrap();
    let processor = ImageProcessor::new(model).await.unwrap(); 

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
