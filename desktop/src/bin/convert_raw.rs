use argh::FromArgs;
use env_logger;

#[derive(FromArgs, PartialEq, Debug)]
/// Convert a camera RAW file to TIFF
struct ConvertRaw {
    #[argh(positional)]
    raw_file: String,
    #[argh(positional)]
    tiff_file: String,
}

fn main() {
    env_logger::init();
    let args: ConvertRaw = argh::from_env();

    backend::convert_raw(args.raw_file, args.tiff_file);
}
