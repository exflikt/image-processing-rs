use std::path::PathBuf;

/// Convinient traits for `image::ImageBuffer`
pub mod image_ext {
    use image::{ImageBuffer, Pixel};
    use std::ops::{Deref, DerefMut};

    pub struct EnumerateSubpixels<'a, P: Pixel + 'a> {
        iter: std::slice::Iter<'a, <P as Pixel>::Subpixel>,
        x: u32,
        y: u32,
        ch: u8,
        width: u32,
        height: u32,
    }

    pub struct EnumerateSubpixelsMut<'a, P: Pixel + 'a> {
        iter_mut: std::slice::IterMut<'a, <P as Pixel>::Subpixel>,
        x: u32,
        y: u32,
        ch: u8,
        width: u32,
        height: u32,
    }

    impl<'a, P: Pixel + 'a> Iterator for EnumerateSubpixels<'a, P> {
        type Item = (u32, u32, u8, <P as Pixel>::Subpixel);

        fn next(&mut self) -> Option<Self::Item> {
            if self.ch >= <P as Pixel>::CHANNEL_COUNT {
                (self.ch, self.x) = (0, self.x + 1);
            }
            if self.x >= self.width {
                (self.x, self.y) = (0, self.y + 1);
            }
            if self.y >= self.height {
                None
            } else {
                let ch = self.ch;
                self.ch += 1;
                self.iter
                    .next()
                    .map(|subpixel| (self.x, self.y, ch, *subpixel))
            }
        }
    }

    impl<'a, P: Pixel + 'a> Iterator for EnumerateSubpixelsMut<'a, P> {
        type Item = (u32, u32, u8, &'a mut <P as Pixel>::Subpixel);

        fn next(&mut self) -> Option<Self::Item> {
            if self.ch >= <P as Pixel>::CHANNEL_COUNT {
                (self.ch, self.x) = (0, self.x + 1);
            }
            if self.x >= self.width {
                (self.x, self.y) = (0, self.y + 1);
            }
            if self.y >= self.height {
                None
            } else {
                let ch = self.ch;
                self.ch += 1;
                self.iter_mut
                    .next()
                    .map(|subpixel| (self.x, self.y, ch, subpixel))
            }
        }
    }

    pub trait ImageBufferExt<P: Pixel> {
        fn get_subpixel(&self, x: u32, y: u32, ch: u8) -> P::Subpixel;

        fn enumerate_subpixels(&self) -> EnumerateSubpixels<P>;
    }

    pub trait ImageBufferExtMut<P: Pixel> {
        fn get_subpixel_mut(&mut self, x: u32, y: u32, ch: u8) -> &mut P::Subpixel;

        fn enumerate_subpixels_mut(&mut self) -> EnumerateSubpixelsMut<P>;
    }

    impl<P, Container> ImageBufferExt<P> for ImageBuffer<P, Container>
    where
        P: Pixel,
        Container: Deref<Target = [<P as Pixel>::Subpixel]>,
    {
        fn get_subpixel(&self, x: u32, y: u32, ch: u8) -> <P as Pixel>::Subpixel {
            assert!(ch <= <P as Pixel>::CHANNEL_COUNT);
            self.get_pixel(x, y).channels()[usize::from(ch)]
        }

        fn enumerate_subpixels(&self) -> EnumerateSubpixels<P> {
            let (width, height) = self.dimensions();
            EnumerateSubpixels {
                iter: self.deref().iter(),
                x: 0,
                y: 0,
                ch: 0,
                width,
                height,
            }
        }
    }

    impl<P, Container> ImageBufferExtMut<P> for ImageBuffer<P, Container>
    where
        P: Pixel,
        Container: Deref<Target = [<P as Pixel>::Subpixel]> + DerefMut,
    {
        fn get_subpixel_mut(&mut self, x: u32, y: u32, ch: u8) -> &mut <P as Pixel>::Subpixel {
            assert!(ch <= <P as Pixel>::CHANNEL_COUNT);
            &mut self.get_pixel_mut(x, y).channels_mut()[usize::from(ch)]
        }

        fn enumerate_subpixels_mut(&mut self) -> EnumerateSubpixelsMut<P> {
            let (width, height) = self.dimensions();
            EnumerateSubpixelsMut {
                iter_mut: self.deref_mut().iter_mut(),
                x: 0,
                y: 0,
                ch: 0,
                width,
                height,
            }
        }
    }
}

pub mod filter {
    use super::image_ext::{ImageBufferExt, ImageBufferExtMut};
    use image::{Pixel, Primitive};
    use num_traits::ToPrimitive;

    pub type Image<P> = image::ImageBuffer<P, Vec<u8>>;

    pub struct FilterViewIndices {
        x: u32,
        y: u32,
        k: u16,
        l: u16,
        k_idx: i32,
        l_idx: i32,
        width: u32,
        height: u32,
    }

    impl Iterator for FilterViewIndices {
        type Item = (u32, u32, (i32, i32));

        fn next(&mut self) -> Option<Self::Item> {
            const fn wrap_within(v: i64, bound: i64) -> i64 {
                match v {
                    v if v < 0 => v + bound,
                    v if bound <= v => v - bound,
                    v => v,
                }
            }

            if self.l_idx > self.l.into() {
                (self.l_idx, self.k_idx) = (-i32::from(self.l), self.k_idx + 1);
            }
            if self.k_idx > self.k.into() {
                None
            } else {
                let x = wrap_within(
                    i64::from(self.x) + i64::from(self.l_idx),
                    i64::from(self.width),
                );
                let y = wrap_within(
                    i64::from(self.y) + i64::from(self.k_idx),
                    i64::from(self.height),
                );
                let l_idx = self.l_idx;
                self.l_idx += 1;
                Some((x as u32, y as u32, (self.k_idx, l_idx)))
            }
        }
    }

    pub struct Filter<'a, P: Pixel<Subpixel = u8>> {
        image: &'a Image<P>,
        k: u16,
        l: u16,
    }

    impl<'a, P: Pixel<Subpixel = u8>> Filter<'a, P> {
        pub fn new(image: &'a Image<P>, k: u16, l: u16) -> Self {
            Filter { image, k, l }
        }

        pub fn filter_view_indices(&self, x: u32, y: u32) -> FilterViewIndices {
            let (width, height) = self.image.dimensions();
            FilterViewIndices {
                x,
                y,
                k: self.k,
                l: self.l,
                k_idx: -i32::from(self.k),
                l_idx: -i32::from(self.l),
                width,
                height,
            }
        }

        /// For each subpixel in the given image, do a fold operation. `f1` receives an
        /// accumulator, a surrounding subpixel, and the position relative to the center of a
        /// kernel matrix. `f1` is executed until all the subpixels in the kernel is applied. `f2`
        /// updates the pixel using the accumulator. Because `init` may be used for multiple times,
        /// it must implement `Clone`.
        ///
        /// See also `subpixels_fold_mut`.
        pub fn subpixels_fold<B, F1, F2>(&self, init: B, mut f1: F1, mut f2: F2) -> Image<P>
        where
            B: Clone,
            F1: FnMut(B, P::Subpixel, (i32, i32)) -> B,
            F2: FnMut(B) -> P::Subpixel,
        {
            let mut filtered = self.image.clone();
            let mut init_vec = vec![init.clone(); P::CHANNEL_COUNT.into()];

            for (i, j, ch, subpixel) in filtered.enumerate_subpixels_mut() {
                let init_ch = &mut init_vec[usize::from(ch)];
                for (x, y, pos) in self.filter_view_indices(i, j) {
                    let og_subpixel = self.image.get_subpixel(x, y, ch);
                    *init_ch = f1(init_ch.clone(), og_subpixel, pos);
                }
                *subpixel = f2(init_ch.clone());
                *init_ch = init.clone();
            }
            filtered
        }

        /// This function is a relaxed version of `subpixels_fold` in that the `f1` and `f2`
        /// closure can modify the `init` values. This comes in handy when the cost of cloning the
        /// `init` values is too expensive. This way we can access the same owned values through
        /// iterations. Note that the previous accumulator value is not initialized so it should be
        /// updated accordingly before the next iteration starts.
        ///
        /// See also `subpixels_fold`.
        pub fn subpixels_fold_mut<B, F1, F2>(
            &self,
            init: &mut [B],
            mut f1: F1,
            mut f2: F2,
        ) -> Image<P>
        where
            F1: FnMut(&mut B, P::Subpixel, (i32, i32)),
            F2: FnMut(&mut P::Subpixel, &mut B),
        {
            assert_eq!(init.len(), P::CHANNEL_COUNT as usize);

            let mut filtered = self.image.clone();
            let accs = init;
            for (i, j, ch, subpixel) in filtered.enumerate_subpixels_mut() {
                for (x, y, pos) in self.filter_view_indices(i, j) {
                    let og_subpixel = self.image.get_subpixel(x, y, ch);
                    f1(&mut accs[usize::from(ch)], og_subpixel, pos);
                }
                f2(subpixel, &mut accs[usize::from(ch)]);
            }
            filtered
        }

        fn conv_1d_filter_reduced(&self, row_kernel: &[f32], col_kernel: &[f32]) -> Image<P> {
            assert_eq!(row_kernel.len(), self.l as usize + 1);
            assert_eq!(col_kernel.len(), self.k as usize + 1);

            let mut sum = 0f32;
            for (l, r) in row_kernel.iter().cloned().enumerate() {
                for (k, c) in col_kernel.iter().cloned().enumerate() {
                    sum += match (k, l) {
                        (k, l) if k == 0 && l == 0 => r * c,
                        (k, l) if k == 0 || l == 0 => 2.0 * (r * c),
                        _ => 4.0 * r * c,
                    };
                }
            }
            let sum = sum;

            self.subpixels_fold(
                P::Subpixel::DEFAULT_MIN_VALUE.to_f32().unwrap(),
                |acc, subpixel, (k, l)| {
                    let (k, l) = (k.unsigned_abs() as u16, l.unsigned_abs() as u16);
                    let coef = row_kernel[l as usize] * col_kernel[k as usize];
                    acc + (f32::from(subpixel) * coef)
                },
                |acc| (acc / sum).to_u8().unwrap(),
            )
        }

        pub fn boxed(&self) -> Image<P> {
            let n: usize = ((2 * self.k as u32 + 1) * (2 * self.l as u32 + 1)) as usize;

            self.subpixels_fold(
                P::Subpixel::DEFAULT_MIN_VALUE.to_usize().unwrap(),
                |acc, sp, _pos| acc + sp.to_usize().unwrap(),
                |acc| (acc / n).to_u8().unwrap(),
            )
        }

        pub fn gaussian(&self, sigma: f32) -> Image<P> {
            let gauss = |x: u16| (-(x.pow(2) as f32) / (2.0 * sigma.powi(2))).exp();
            self.conv_1d_filter_reduced(
                Vec::from_iter((0..=self.l).map(gauss)).as_slice(),
                Vec::from_iter((0..=self.k).map(gauss)).as_slice(),
            )
        }

        pub fn minimum(&self) -> Image<P> {
            self.subpixels_fold(
                P::Subpixel::DEFAULT_MAX_VALUE,
                |min_sp, sp, _pos| (sp < min_sp).then_some(sp).unwrap_or(min_sp),
                std::convert::identity,
            )
        }

        pub fn maximum(&self) -> Image<P> {
            self.subpixels_fold(
                P::Subpixel::DEFAULT_MIN_VALUE,
                |max_sp, sp, _pos| (sp > max_sp).then_some(sp).unwrap_or(max_sp),
                std::convert::identity,
            )
        }

        pub fn median(&self) -> Image<P> {
            let len = (2 * self.k as usize + 1) * (2 * self.l as usize + 1);

            self.subpixels_fold_mut(
                vec![vec![P::Subpixel::DEFAULT_MIN_VALUE; len]; P::CHANNEL_COUNT.into()]
                    .as_mut_slice(),
                |v, subpixel, (k, l)| {
                    let idx = (k + self.k as i32) * (2 * self.l as i32 + 1) + (l + self.l as i32);
                    v[idx as usize] = subpixel;
                },
                |new_subpixel, v| {
                    v.sort_unstable();
                    *new_subpixel = v[len / 2 + 1];
                },
            )
        }
    }
}

enum Command {
    Box,
    GaussianBlur,
    Minimum,
    Maximum,
    Median,
}

impl TryFrom<String> for Command {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "box" => Ok(Command::Box),
            "gauss" => Ok(Command::GaussianBlur),
            "min" => Ok(Command::Minimum),
            "max" => Ok(Command::Maximum),
            "med" => Ok(Command::Median),
            _ => Err("Command should be box, gauss, max, min, or med".to_string()),
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).take(2);
    let command: Command = args.next().expect("No arguments provided").try_into()?;
    let input_path: PathBuf = args.next().expect("No file argument provided").into();

    let img = image::open(input_path.as_path())?.into_luma8();

    let filter = filter::Filter::new(&img, 2, 2);
    match command {
        Command::Box => {
            let filtered = filter.boxed();
            let output_path = postfix_path(input_path, "_box");
            filtered.save(output_path)?
        }
        Command::GaussianBlur => {
            let sigmas = [0.5, 0.75, 1.0, 1.25, 10.0, 150.0];
            for sigma in sigmas {
                let filtered = filter.gaussian(sigma);
                println!("Filtered sigma = {}", sigma);
                let postfix = format!("_gauss_{}", sigma);
                let output_path = postfix_path(input_path.clone(), &postfix);
                filtered.save(output_path)?
            }
        }
        Command::Minimum => {
            let filtered = filter.minimum();
            let output_path = postfix_path(input_path, "_min");
            filtered.save(output_path)?
        }
        Command::Maximum => {
            let filtered = filter.maximum();
            let output_path = postfix_path(input_path, "_max");
            filtered.save(output_path)?
        }
        Command::Median => {
            let filtered = filter.median();
            let output_path = postfix_path(input_path, "_med");
            filtered.save(output_path)?
        }
    }
    Ok(())
}
