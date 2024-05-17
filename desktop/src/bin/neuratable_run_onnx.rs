use std::str::FromStr;

use argh::FromArgs;
use backend::image_processor::{ImageColorModel, ImageProcessor};
use backend::model_value_range::ModelValueRange;
use std::path::Path;

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
    /// if enabled, input_image and output_image should be directories and NeuraTable will process
    /// all images in the input directory to a file in the output directory
    #[argh(switch, short = 'b')]
    batch_process: bool,
    /// a string that will be appended to the filename of batch-processed files
    #[argh(option, short = 's')]
    batch_process_output_suffix: Option<String>,
    /// the value range for input values. Can be a positive float number for [0,x] ranges or "+-x"
    /// for [-x,x] ranges
    #[argh(option, default = "ModelValueRange::asymmetric(1.0)")]
    input_range: ModelValueRange,
    #[argh(option, default = "ModelValueRange::asymmetric(1.0)")]
    /// the value range for output values. Can be a positive float number for [0,x] ranges or "+-x"
    /// for [-x,x] ranges
    output_range: ModelValueRange,
}

async fn run(args: RunOnnx) {
    let mut r = std::fs::File::open(&args.onnx_model).unwrap();

    let runner = backend::model_runner::ModelRunner::new(&mut r, args.force_cpu)
        .await
        .unwrap();

    let mut processor = ImageProcessor::new(
        runner,
        args.model_channel_order.0,
        args.input_range,
        args.output_range,
    )
    .await
    .unwrap();

    if !args.batch_process {
        let input_image = image::open(&args.input_image).unwrap().to_rgb16();
        let output_image = processor.process_image(input_image).await.unwrap();

        // FIXME: For JPG Output, we need to scale the image data back to 8 Bit RGB
        // We need to find a generic way to solve this issue
        output_image.save(&args.output_image).unwrap();
    } else {
        let input_dir = Path::new(&args.input_image);
        let output_dir = Path::new(&args.output_image);
        if !output_dir.is_dir() {
            panic!("Output directory path is not a directory!");
        }
        for maybe_entry in input_dir
            .read_dir()
            .expect("Could not read input directory")
        {
            if let Ok(entry) = maybe_entry {
                if entry.path().is_file() {
                    // TODO: We need to check if this is actually an image!
                    let input_image = image::open(entry.path()).unwrap().to_rgb16();
                    let output_image_filename =
                        if let Some(suffix) = &args.batch_process_output_suffix {
                            format!(
                                "{}{}.{}",
                                entry.path().file_stem().unwrap().to_string_lossy(),
                                suffix,
                                entry.path().extension().unwrap().to_string_lossy()
                            )
                        } else {
                            entry
                                .path()
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                                .to_string()
                        };
                    let output_image_path = output_dir.join(output_image_filename);
                    let output_image = processor.process_image(input_image).await.unwrap();
                    output_image.save(output_image_path).unwrap();
                }
            }
        }
    }
}

fn main() {
    env_logger::init();
    log::debug!("Test");
    let args: RunOnnx = argh::from_env();
    pollster::block_on(run(args));
}
