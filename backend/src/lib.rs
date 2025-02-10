pub mod image_chunk_iterator;
pub mod image_processor;
pub mod model_runner;
pub mod model_value_range;

mod chunksize;
use regex::Regex;
use std::io::Write;
use std::path::Path;
use std::process::Command;

pub use chunksize::ChunkSize;

pub fn convert_raw(raw_path: impl AsRef<Path>, tiff_path: impl AsRef<Path>) {
    let xml_data = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
                            <x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
                                <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
                                    <rdf:Description rdf:about=""
                                        xmlns:exif="http://ns.adobe.com/exif/1.0/"
                                        xmlns:xmp="http://ns.adobe.com/xap/1.0/"
                                        xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
                                        xmlns:darktable="http://darktable.sf.net/"
                                        exif:DateTimeOriginal="2023:08:20 12:49:58.000"
                                        xmpMM:DerivedFrom="{}">
                                    </rdf:Description>
                                </rdf:RDF>
                            </x:xmpmeta>"#,
        raw_path.as_ref().to_string_lossy()
    );

    let mut xml_file = tempfile::NamedTempFile::new().unwrap();
    xml_file.write_all(xml_data.as_bytes()).unwrap();

    log::info!("Running darktable to convert raw file...");
    let output = Command::new("darktable-cli")
        .arg(raw_path.as_ref())
        .arg(xml_file.path())
        .arg(tiff_path.as_ref())
        .output()
        .expect("failed to execute process");
    let stdout = String::from_utf8(output.stdout).expect("Darktable output is not valid UTF-8!");
    let stderr = String::from_utf8(output.stderr).expect("Darktable output is not valid UTF-8!");

    log::info!("Darktable output: {}", stdout);
    if !stderr.is_empty() {
        log::error!("Darktable stderr: {}", stderr)
    }

    let re = Regex::new(r"exported to `(.*)'").unwrap();
    let cap = re
        .captures_iter(&stdout)
        .next()
        .expect("Could not parse darktable-cli output.");
    let export_name = cap.get(1).unwrap();
    std::fs::rename(export_name.as_str(), tiff_path).expect("Failed to rename file");
}
