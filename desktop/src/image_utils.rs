use std::{ffi::OsStr, path::Path, process::Command};

use image::buffer::ConvertBuffer;

type NeuratableImage = image::ImageBuffer<image::Rgb<u16>, Vec<u16>>;

pub fn load_image<T: AsRef<Path>>(p: T) -> NeuratableImage {
    image::open(&p)
        .unwrap_or_else(|_| {
            log::warn!(
                "Trying darktable conversion for {}",
                p.as_ref().to_string_lossy()
            );
            let tiff_file = tempfile::NamedTempFile::with_suffix(".tif").unwrap();
            backend::convert_raw(p, tiff_file.path());

            image::open(tiff_file.path()).unwrap()
        })
        .to_rgb16()
}

pub fn save_image(image: &NeuratableImage, path: impl AsRef<Path>) {
    match image.save(&path) {
        Err(image::ImageError::Unsupported(_)) => {
            // If we cannot write the image in the 16 bit representation, we try scaling to 8 bit RGB instead.
            ConvertBuffer::<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>>::convert(image)
                .save(&path)
                .expect("Could not write image")
        }
        Err(e) => Err(e).expect("Could not write image"),
        _ => (),
    };
}

pub struct MetadataHandler {
    has_exiftool: bool,
}

impl MetadataHandler {
    pub fn new() -> Self {
        let has_exiftool = Command::new("exiftool").arg("-ver").output().is_ok();
        if !has_exiftool {
            log::error!(
                "exiftool could not be executed! Image metadata will be lost after processing!"
            )
        }
        Self { has_exiftool }
    }

    pub fn copy_metadata<T: AsRef<Path>, U: AsRef<Path>>(&self, source: T, destination: U) {
        if self.has_exiftool {
            if Command::new("exiftool")
                .args([
                    OsStr::new("-overwrite_original"),
                    OsStr::new("-tagsFromFile"),
                    source.as_ref().as_os_str(),
                    destination.as_ref().as_os_str(),
                ])
                .output()
                .is_err()
            {
                log::error!(
                    "Failed to run exiftool for {}",
                    source.as_ref().to_string_lossy()
                );
            }
        }
    }
}
