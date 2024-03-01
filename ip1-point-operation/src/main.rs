use histogram::Histogram;
use std::path::PathBuf;

pub mod histogram {
    use image::GrayImage;
    use std::ops::Range;
    use std::path::Path;

    pub const HIST_SIZE: u8 = u8::MAX;

    /// This implementation does not prevent integer overflow/underflows potentially caused by
    /// exessive increment/decrements.
    #[derive(Debug, Clone)]
    pub struct Histogram {
        v: [usize; HIST_SIZE as usize + 1],
    }

    impl Default for Histogram {
        fn default() -> Self {
            Self {
                v: [0; HIST_SIZE as usize + 1],
            }
        }
    }

    impl Histogram {
        pub fn new(img: &GrayImage) -> Self {
            Histogram::calc_hist_from_img(img)
        }

        pub fn calc_hist_from_img(img: &GrayImage) -> Self {
            let mut hist: Histogram = Default::default();
            for intensity in img.pixels().flat_map(|p| p.0) {
                hist.increment(intensity);
            }
            hist
        }

        pub fn increment(&mut self, index: u8) {
            *self.get_mut(index) += 1;
        }

        pub fn get(&self, index: u8) -> usize {
            // SAFETY: `idx` is `u8` so it never reaches over the 255th item
            unsafe { *self.v.get_unchecked(usize::from(index)) }
        }

        pub fn get_mut(&mut self, index: u8) -> &mut usize {
            // SAFETY: `idx` is `u8` so it never reaches over the 255th item
            unsafe { self.v.get_unchecked_mut(usize::from(index)) }
        }

        pub fn max(&self) -> usize {
            let res = self.v.into_iter().reduce(usize::max);
            // SAFETY: `reduce` returns `None` if the iterator is empty, which is not the case
            //         because the array has a fixed size of 256.
            unsafe { res.unwrap_unchecked() }
        }

        pub fn norm_hist(&mut self, Range { start, end }: Range<usize>) {
            let hist_max = self.max();
            for p in self.v.iter_mut() {
                *p = start + (*p as f64 * (end - start) as f64 / hist_max as f64).ceil() as usize;
            }
        }

        pub fn cumulate(&mut self) {
            for i in 1..HIST_SIZE {
                *self.get_mut(i) += self.get(i - 1);
            }
        }

        /// Generate a histogram image in grayscale. Image size is 400x512. This method clones the
        /// histogram for normalization before creating the image.
        pub fn gen_histimg(&self) -> GrayImage {
            const HEIGHT: u32 = 400;
            const WIDTH: u32 = 512; // = (HIST_SIZE + 1) * 2
            const BAR_WIDTH: u32 = 2; // = WIDTH / (HIST_SIZE + 1)

            let mut hist = self.clone();
            hist.norm_hist(0..HEIGHT as usize);

            let mut img = GrayImage::from_pixel(WIDTH, HEIGHT, [255].into());
            for (i, p) in hist.v.iter().enumerate() {
                let p1 = (i as u32 * BAR_WIDTH, HEIGHT);
                let p2 = ((i + 1) as u32 * BAR_WIDTH, HEIGHT - *p as u32);
                draw_histbar(&mut img, p1, p2);
            }
            img
        }

        pub fn save<P>(&self, path: P) -> image::ImageResult<()>
        where
            P: AsRef<Path>,
        {
            self.gen_histimg().save(path)
        }

        /// Note: this method mutate the histogram internally.
        pub fn save_cumul<P>(&mut self, path: P) -> image::ImageResult<()>
        where
            P: AsRef<Path>,
        {
            self.cumulate();
            self.gen_histimg().save(path)
        }
    }

    fn draw_histbar(img: &mut GrayImage, (x1, y1): (u32, u32), (x2, y2): (u32, u32)) {
        let (x1, x2) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
        let (y1, y2) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
        for y in y1..=y2 {
            for x in x1..=x2 {
                if let Some(p) = img.get_pixel_mut_checked(x, y) {
                    *p = [0].into();
                }
            }
        }
    }
}

pub mod point_operation {
    pub const fn invert(intensity: u8) -> u8 {
        // The unary logical negation operator is effectively an pixel inversion.
        !intensity // i.e. `std::ops::Not::not(intensity)`
    }

    pub struct Binarizer {
        pub thresh: u8,
        pub min: u8,
        pub max: u8,
    }

    impl Binarizer {
        pub fn binarize(&self, intensity: u8) -> u8 {
            if intensity < self.thresh {
                self.min
            } else {
                self.max
            }
        }
    }

    pub struct Equalizer {
        pub hist: super::histogram::Histogram,
        pub width: u32,
        pub height: u32,
    }

    impl Equalizer {
        pub fn equalize(&self, intensity: u8) -> u8 {
            (self.hist.get(intensity) as f64 * f64::from(u8::MAX)
                / (f64::from(self.width) * f64::from(self.height)))
            .ceil() as u8
        }
    }

    pub fn gamma_correct(intensity: u8, gamma: f64) -> u8 {
        ((f64::from(intensity) / f64::from(u8::MAX)).powf(gamma) * f64::from(u8::MAX)).ceil() as u8
    }
}

enum Command {
    Inv,
    Bin,
    Eq,
    Gc,
}

impl TryFrom<String> for Command {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "inv" => Ok(Command::Inv),
            "bin" => Ok(Command::Bin),
            "eq" => Ok(Command::Eq),
            "gc" => Ok(Command::Gc),
            _ => Err("Command should be inv, bin, eq, or gc".to_string()),
        }
    }
}

// Postfix `&str` into filename
fn postfix_path(pathbuf: PathBuf, postfix: &str) -> PathBuf {
    let parent = pathbuf.parent().unwrap_or_else(|| "".as_ref());
    let mut file_name = pathbuf.file_stem().expect("Empty file name").to_os_string();
    file_name.push(std::ffi::OsString::from(postfix));
    file_name.push(".");
    file_name.push(pathbuf.extension().unwrap_or_default());
    [parent, file_name.as_ref()].iter().collect()
}

fn apply<F: Fn(u8) -> u8>(img: &mut image::GrayImage, f: F) {
    img.pixels_mut()
        .flat_map(|p| &mut p.0)
        .for_each(|intensity| *intensity = f(*intensity))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).take(2);
    let command: Command = args.next().expect("No arguments provided").try_into()?;
    let input_path: PathBuf = args.next().expect("No file argument provided").into();

    let mut img = image::open(input_path.as_path())?.into_luma8();

    match command {
        Command::Inv => {
            let hist = Histogram::new(&img);
            hist.save(postfix_path(input_path.clone(), "_hist"))?;

            apply(&mut img, point_operation::invert);

            let output_path = postfix_path(input_path, "_inv");
            img.save(output_path.clone())?;
            let hist = Histogram::new(&img);
            hist.save(postfix_path(output_path, "_hist"))?;
        }
        Command::Bin => {
            let hist = Histogram::new(&img);
            hist.save(postfix_path(input_path.clone(), "_hist"))?;

            let binarizer = point_operation::Binarizer {
                thresh: 128_u8,
                min: 0x00_u8,
                max: 0xff_u8,
            };
            apply(&mut img, |v| binarizer.binarize(v));

            let output_path = postfix_path(input_path, "_bin");
            img.save(output_path.clone())?;
            let hist = Histogram::new(&img);
            hist.save(postfix_path(output_path, "_hist"))?;
        }
        Command::Eq => {
            let mut hist = Histogram::new(&img);
            hist.save_cumul(postfix_path(input_path.clone(), "_cumul_hist"))?;

            let (width, height) = img.dimensions();
            let equalizer = point_operation::Equalizer {
                hist,
                width,
                height,
            };
            apply(&mut img, |v| equalizer.equalize(v));

            let output_path = postfix_path(input_path, "_eq");
            img.save(output_path.clone())?;
            let mut hist = Histogram::new(&img);
            hist.save_cumul(postfix_path(output_path, "_cumul_hist"))?;
        }
        Command::Gc => {
            let hist = Histogram::new(&img);
            hist.save(postfix_path(input_path.clone(), "_hist"))?;

            let gammas = [0.25, 0.5, 1.0, 2.0];
            for gamma in gammas {
                let mut img = img.clone();
                apply(&mut img, |v| point_operation::gamma_correct(v, gamma));

                let output_path = postfix_path(input_path.clone(), &format!("_{gamma}_gc"));
                img.save(output_path.clone())?;
                let hist = Histogram::new(&img);
                hist.save(postfix_path(output_path, "_hist"))?;
            }
        }
    }
    Ok(())
}
