use std::path::PathBuf;

pub mod conv {
    use image::{GrayImage, Rgb, RgbImage};
    use num_traits::ToPrimitive;

    /// The HSV Color Pixel: a named tuple of hue, saturation, and value.
    ///
    /// Hue and saturation are assumed to be in the range of:
    /// $H \in [0.0, 360.0], S \in [0.0, 1.0]$
    pub struct Hsv((f32, f32, u8));

    fn rgb2hsv(Rgb([r, g, b]): Rgb<u8>) -> Hsv {
        let (max, min) = (r.max(g).max(b), r.min(g).min(b));
        let v = max;
        let [r, g, b, max, min] = [r, g, b, max, min].map(f32::from);
        let s = if max != 0. { (max - min) / max } else { 0. };
        let h = match min {
            min if min == max => 0.,
            min if min == b => 60. * (g - r) / (max - min) + 60.,
            min if min == r => 60. * (b - g) / (max - min) + 120.,
            min if min == g => 60. * (r - b) / (max - min) + 300.,
            _ => unreachable!(),
        };
        Hsv((h, s, v))
    }

    fn rgb2gray(Rgb(rgb_array): Rgb<u8>) -> u8 {
        const W_R: f32 = 0.299;
        const W_G: f32 = 0.587;
        const W_B: f32 = 0.114;
        let [r, g, b] = rgb_array.map(f32::from);
        (W_R * r + W_G * g + W_B * b).to_u8().unwrap()
    }

    pub trait FromRgb {
        fn to_hsv_vec(&self) -> Vec<Hsv>;

        fn to_hsv_images(&self) -> [GrayImage; 3];

        fn to_gray(&self) -> GrayImage;
    }

    impl FromRgb for RgbImage {
        fn to_hsv_vec(&self) -> Vec<Hsv> {
            self.pixels().copied().map(rgb2hsv).collect()
        }

        fn to_hsv_images(&self) -> [GrayImage; 3] {
            let (width, height) = self.dimensions();
            let hsv = self.to_hsv_vec();
            let mut h_img = GrayImage::new(width, height);
            let mut s_img = GrayImage::new(width, height);
            let mut v_img = GrayImage::new(width, height);
            for (i, &Hsv((h, s, v))) in hsv.iter().enumerate() {
                h_img.as_mut()[i] = (f32::from(u8::MAX) * h / 360.).to_u8().unwrap();
                s_img.as_mut()[i] = (f32::from(u8::MAX) * s).to_u8().unwrap();
                v_img.as_mut()[i] = v;
            }
            [h_img, s_img, v_img]
        }

        fn to_gray(&self) -> GrayImage {
            let (width, height) = self.dimensions();
            GrayImage::from_raw(
                width,
                height,
                self.pixels().copied().map(rgb2gray).collect(),
            )
            .unwrap()
        }
    }
}

enum Command {
    Rgb2Hsv,
    Rgb2Gray,
}

impl TryFrom<String> for Command {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "rgb2hsv" => Ok(Command::Rgb2Hsv),
            "rgb2gray" => Ok(Command::Rgb2Gray),
            _ => Err("Command should be either rgb2hsv or rgb2gray".to_string()),
        }
    }
}

// Postfix `&str` into filename
fn postfix_path(pathbuf: PathBuf, postfix: &str) -> PathBuf {
    let parent = pathbuf.parent().unwrap_or_else(|| "".as_ref());
    let mut file_name = pathbuf.file_stem().expect("Empty file name").to_os_string();
    file_name.push(postfix);
    file_name.push(".");
    file_name.push(pathbuf.extension().unwrap_or_default());
    [parent, file_name.as_ref()].iter().collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use conv::FromRgb;

    let mut args = std::env::args().skip(1).take(2);
    let command: Command = args.next().expect("No arguments provided").try_into()?;
    let input_path: PathBuf = args.next().expect("No file argument provided").into();
    let img = image::open(input_path.as_path())?.into_rgb8();

    match command {
        Command::Rgb2Hsv => {
            let [h, s, v] = img.to_hsv_images();
            h.save(postfix_path(input_path.clone(), "_hue"))?;
            s.save(postfix_path(input_path.clone(), "_sat"))?;
            v.save(postfix_path(input_path, "_val"))?;
        }
        Command::Rgb2Gray => {
            img.to_gray().save(postfix_path(input_path, "_gray"))?;
        }
    }
    Ok(())
}
