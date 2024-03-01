use image::codecs::bmp::BmpDecoder;
use image::error::{LimitError, LimitErrorKind, ParameterError, ParameterErrorKind};
use image::{ColorType, DynamicImage, ImageBuffer, ImageDecoder, ImageError, ImageResult};

#[allow(dead_code)]
fn read_image<P>(path: P) -> image::ImageResult<image::DynamicImage>
where
    P: AsRef<std::path::Path>,
{
    fn decoder_to_vec<'a>(decoder: impl ImageDecoder<'a>) -> ImageResult<Vec<u8>> {
        let total_bytes = usize::try_from(decoder.total_bytes());
        if total_bytes.is_err() || total_bytes.unwrap() > isize::max_value() as usize {
            return Err(ImageError::Limits(LimitError::from_kind(
                LimitErrorKind::InsufficientMemory,
            )));
        }
        let mut buf = vec![u8::MIN; total_bytes.unwrap()];
        decoder.read_image(buf.as_mut_slice())?;
        Ok(buf)
    }

    let file = std::fs::File::open(path).map_err(ImageError::IoError)?;
    let buffered_read = std::io::BufReader::new(file);
    let mut decoder = BmpDecoder::new(buffered_read)?;

    // Workaround for the BMP decoder wrongly(?) treating a gray image as colored.
    // https://github.com/image-rs/image/pull/1572
    if decoder.get_palette().is_some() {
        decoder.set_indexed_color(true);
    }

    let (w, h) = decoder.dimensions();
    let color_type = decoder.color_type();
    let buf = decoder_to_vec(decoder)?;

    let image = match color_type {
        ColorType::L8 => ImageBuffer::from_raw(w, h, buf).map(DynamicImage::ImageLuma8),
        ColorType::Rgb8 => ImageBuffer::from_raw(w, h, buf).map(DynamicImage::ImageRgb8),
        color_type => panic!("Unsupported format: {:?}", color_type),
    };
    image.ok_or_else(|| {
        ImageError::Parameter(ParameterError::from_kind(
            ParameterErrorKind::DimensionMismatch,
        ))
    })
}
