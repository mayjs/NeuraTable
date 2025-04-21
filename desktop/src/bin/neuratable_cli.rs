use backend::image_processor::{ImageColorModel, ImageProcessor};
use backend::model_value_range::ModelValueRange;
use clap::error::ErrorKind;
use clap::{Args, CommandFactory, Parser, Subcommand};
use desktop::image_utils::{load_image, save_image, MetadataHandler};
use std::io::Cursor;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Parser)]
#[command(name = "neuratable-cli")]
#[command(about = "Neuratable - Using neural networks for digital photography procesing", long_about = None)]
struct NeuratableCli {
    #[command(subcommand)]
    command: NeuratableCommands,
}

#[derive(Debug, Subcommand)]
enum NeuratableCommands {
    /// denoise an image with nind-denoise
    Denoise(ImageProcessingArgs),
}

#[derive(Debug, Args)]
#[command(flatten_help = true)]
struct ImageProcessingArgs {
    #[arg(value_name = "IMAGE", required = true)]
    /// paths for the input images
    images: Vec<String>,
    #[arg(value_name = "OUTPUT_PATTERN")]
    /// pattern for the output images, %NAME% will be replaced with the input image filename without extension
    output_pattern: String,
}

impl NeuratableCommands {
    fn get_image_processing_args(&self) -> Option<&ImageProcessingArgs> {
        match self {
            NeuratableCommands::Denoise(image_processing_args) => Some(image_processing_args),
        }
    }
}

impl ImageProcessingArgs {
    fn validate(&self) {
        if self.images.len() > 1 && !self.output_pattern.contains("%NAME%") {
            let mut cmd = NeuratableCli::command();
            cmd.error(
                ErrorKind::ValueValidation,
                "OUTPUT_PATTERN must include %NAME% if multiple IMAGE entries are used",
            )
            .exit();
        }
    }
}

fn render_output_pattern(input: &str, output_pattern: &str) -> String {
    // FIXME: Don't panic here, propagate errors instead
    let input_path = PathBuf::from_str(input).expect("Input is not a valid path");
    let input_name = input_path
        .with_extension("")
        .file_name()
        .expect("File has no filename")
        .to_str()
        .expect("File path cannot be represented as UTF-8")
        .to_owned();
    let output = output_pattern.replace("%NAME%", &input_name);
    let p = PathBuf::from_str(&output).expect("Generated output is not a valid path");
    if p.extension().is_none() {
        let input_extension = input_path
            .extension()
            .expect("Input path has no extension")
            .to_str()
            .expect("File path cannot be represented as UTF-8");
        p.with_extension(input_extension)
            .to_str()
            .expect("File path cannot be represented as UTF-8")
            .to_owned()
    } else {
        output
    }
}

trait ProcessingTask {
    async fn run(&mut self, input: &str, output: &str);
}

struct OnnxModelProcessingTask {
    processor: ImageProcessor,
    metadata_handler: MetadataHandler,
}

impl OnnxModelProcessingTask {
    async fn new(model: &[u8]) -> Self {
        let runner = backend::model_runner::ModelRunner::new(&mut Cursor::new(model), false)
            .await
            .unwrap();

        // FIXME: Color model and value ranges should not be hard-coded here. This might be an issue if we add new models in the future.
        let processor = ImageProcessor::new(
            runner,
            ImageColorModel::RGB,
            ModelValueRange::asymmetric(1.0),
            ModelValueRange::asymmetric(1.0),
        )
        .await
        .unwrap();

        let metadata_handler = MetadataHandler::new();
        Self {
            processor,
            metadata_handler,
        }
    }
}

impl ProcessingTask for OnnxModelProcessingTask {
    async fn run(&mut self, input: &str, output: &str) {
        let input_image = load_image(input);
        let output_image = self.processor.process_image(input_image).await.unwrap();

        save_image(&output_image, output);
        self.metadata_handler.copy_metadata(input, output);
    }
}

async fn run_async(args: NeuratableCli) {
    let mut processing_task = match &args.command {
        NeuratableCommands::Denoise(_) => {
            OnnxModelProcessingTask::new(include_bytes!(
                "../../../models/denoise_unet_no_batch-sim.onnx"
            ))
            .await
        }
    };

    if let Some(image_processing_args) = &args.command.get_image_processing_args() {
        image_processing_args.validate();

        println!(
            "Starting image processing for {} images...",
            image_processing_args.images.len()
        );

        for (input, output) in image_processing_args
            .images
            .iter()
            .zip(std::iter::repeat(&image_processing_args.output_pattern))
            .map(|(input, output_pattern)| (input, render_output_pattern(input, output_pattern)))
        {
            processing_task.run(input, &output).await;
            println!("Done: {} -> {}", input, output);
        }
    }
}

fn main() {
    env_logger::init();
    let args = NeuratableCli::parse();

    pollster::block_on(run_async(args));
}
