use std::str::FromStr;

use argh::FromArgs;
use backend::image_processor::{ImageColorModel, ImageProcessor};
use backend::model_value_range::ModelValueRange;
use std::path::Path;
use std::process::Command;

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
    /// if enabled, batch processing will only consider images where the output image does not exist
    #[argh(switch, short = 'n')]
    no_overwrite: bool,
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

    let has_exiftool = Command::new("exiftool").arg("-ver").output().is_ok();
    if !has_exiftool {
        log::error!("exiftool could not be executed! Image metadata will be lost after processing!")
    }
    let copy_metadata = |source: &str, destination: &str| -> () {
        if has_exiftool {
            if Command::new("exiftool")
                .args(["-overwrite_original", "-tagsFromFile", source, destination])
                .output()
                .is_err()
            {
                log::error!("Failed to run exiftool for {}", source);
            }
        }
    };

    if !args.batch_process {
        let input_image = image::open(&args.input_image).unwrap().to_rgb16();
        let output_image = processor.process_image(input_image).await.unwrap();

        // FIXME: For JPG Output, we need to scale the image data back to 8 Bit RGB
        // We need to find a generic way to solve this issue
        output_image.save(&args.output_image).unwrap();
        copy_metadata(&args.input_image, &args.output_image);
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
                    // TODO: We need to check if the input is actually an image!
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
                    if !args.no_overwrite || !output_image_path.exists() {
                        let input_image = image::open(entry.path()).unwrap().to_rgb16();
                        let output_image = processor.process_image(input_image).await.unwrap();
                        output_image.save(&output_image_path).unwrap();

                        copy_metadata(
                            entry.path().to_string_lossy().as_ref(),
                            output_image_path.to_string_lossy().as_ref(),
                        )
                    } else {
                        log::info!(
                            "Skipping {} since the output file for it already exists.",
                            entry.path().to_string_lossy()
                        );
                    }
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
