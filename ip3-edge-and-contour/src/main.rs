use std::path::PathBuf;

pub mod filter {
    use image::{GrayImage, ImageBuffer, Luma, Primitive};
    use num_traits::ToPrimitive;

    type LumaImage<P> = ImageBuffer<Luma<P>, Vec<P>>;

    #[derive(Debug, Copy, Clone)]
    pub struct FilterView {
        pub l: u16,
        pub k: u16,
    }

    impl FilterView {
        pub const fn new(l: u16, k: u16) -> Self {
            Self { l, k }
        }

        fn indices(&self, x: u32, y: u32, (width, height): (u32, u32)) -> FilterViewIndices {
            FilterViewIndices {
                x,
                y,
                view: *self,
                l_idx: -i32::from(self.l),
                k_idx: -i32::from(self.k),
                width,
                height,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct FilterViewIndices {
        x: u32,
        y: u32,
        view: FilterView,
        l_idx: i32,
        k_idx: i32,
        width: u32,
        height: u32,
    }

    impl Iterator for FilterViewIndices {
        type Item = (u32, u32, (i32, i32));

        fn next(&mut self) -> Option<Self::Item> {
            if self.l_idx > self.view.l.into() {
                (self.l_idx, self.k_idx) = (-i32::from(self.view.l), self.k_idx + 1);
            }
            if self.k_idx > self.view.k.into() {
                None
            } else {
                let x = add_wrap_within(self.x, self.l_idx, self.width);
                let y = add_wrap_within(self.y, self.k_idx, self.height);
                let l_idx = self.l_idx;
                self.l_idx += 1;
                Some((x, y, (l_idx, self.k_idx)))
            }
        }
    }

    pub struct Kernel<P: Primitive> {
        data: Vec<P>,
        view: FilterView,
    }

    impl<P: Primitive> Kernel<P> {
        pub fn new(data: Vec<P>, l: u16, k: u16) -> Self {
            let data_len = data.len();
            let kernel = Kernel {
                data,
                view: FilterView::new(l, k),
            };
            assert_eq!(data_len, (kernel.rows() * kernel.cols()) as usize);
            kernel
        }

        pub const fn shift_k(&self, k: i32) -> u32 {
            (k + self.view.k as i32) as u32
        }

        pub const fn shift_l(&self, l: i32) -> u32 {
            (l + self.view.l as i32) as u32
        }

        pub const fn rows(&self) -> u32 {
            2 * self.view.k as u32 + 1
        }

        pub const fn cols(&self) -> u32 {
            2 * self.view.l as u32 + 1
        }

        pub fn get_subpixel(&self, l: i32, k: i32) -> &P {
            assert!(l.unsigned_abs() <= self.view.l.into());
            assert!(k.unsigned_abs() <= self.view.k.into());
            let idx = self.shift_k(k) * self.cols() + self.shift_l(l);
            &self.data[idx.to_usize().unwrap()]
        }
    }

    impl<P: Primitive> std::ops::Index<(i32, i32)> for Kernel<P> {
        type Output = P;

        fn index(&self, (l, k): (i32, i32)) -> &Self::Output {
            self.get_subpixel(l, k)
        }
    }

    pub trait FilterExt<P: Primitive, Q: Primitive> {
        fn filter(&self, kernel: Kernel<Q>) -> LumaImage<Q>;

        fn fold_each_view<B, F1, F2>(
            &self,
            init: B,
            view: FilterView,
            f1: F1,
            f2: F2,
        ) -> LumaImage<Q>
        where
            B: Clone,
            F1: FnMut(B, P, (i32, i32)) -> B,
            F2: FnMut(B) -> Q;
    }

    impl<P: Primitive, Q: Primitive> FilterExt<P, Q> for LumaImage<P> {
        fn filter(&self, kernel: Kernel<Q>) -> LumaImage<Q> {
            self.fold_each_view(
                Q::zero(),
                kernel.view,
                |acc, subpixel, pos| acc + Q::from(subpixel).unwrap() * kernel[pos],
                std::convert::identity,
            )
        }

        fn fold_each_view<B, F1, F2>(
            &self,
            init: B,
            view: FilterView,
            mut f1: F1,
            mut f2: F2,
        ) -> LumaImage<Q>
        where
            B: Clone,
            F1: FnMut(B, P, (i32, i32)) -> B,
            F2: FnMut(B) -> Q,
        {
            let mut acc = init.clone();
            let dim = self.dimensions();
            let mut filtered = LumaImage::new(dim.0, dim.1);
            for (x, y, Luma([subpixel])) in filtered.enumerate_pixels_mut() {
                for (view_x, view_y, pos) in view.indices(x, y, dim) {
                    let og_pixel = self[(view_x, view_y)];
                    acc = f1(acc.clone(), og_pixel[0], pos);
                }
                *subpixel = f2(acc.clone());
                acc = init.clone();
            }
            filtered
        }
    }

    fn conv_1d_filter_reduced(
        image: &GrayImage,
        view: FilterView,
        row_kernel: &[f32],
        col_kernel: &[f32],
    ) -> GrayImage {
        assert_eq!(row_kernel.len(), view.l as usize + 1);
        assert_eq!(col_kernel.len(), view.k as usize + 1);

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

        image.fold_each_view(
            0f32,
            view,
            |acc, subpixel, (l, k)| {
                let (l, k) = (l.unsigned_abs() as usize, k.unsigned_abs() as usize);
                let coef = row_kernel[l] * col_kernel[k];
                acc + (f32::from(subpixel) * coef)
            },
            |acc| (acc / sum).to_u8().unwrap(),
        )
    }

    pub fn gaussian(image: &GrayImage, view: FilterView, sigma: f32) -> GrayImage {
        let gauss = |x: u16| (-(x.pow(2) as f32) / (2.0 * sigma.powi(2))).exp();
        conv_1d_filter_reduced(
            image,
            view,
            Vec::from_iter((0..=view.l).map(gauss)).as_slice(),
            Vec::from_iter((0..=view.k).map(gauss)).as_slice(),
        )
    }

    fn fdr_fx_raw(image: &GrayImage) -> LumaImage<i16> {
        image.filter(Kernel::new(vec![-1, 0, 1], 1, 0))
    }

    fn fdr_fy_raw(image: &GrayImage) -> LumaImage<i16> {
        image.filter(Kernel::new(vec![-1, 0, 1], 0, 1))
    }

    pub fn fdr_fx(image: &GrayImage) -> GrayImage {
        let f = |subpixel: &i16| ((subpixel + i16::from(u8::MAX)) / 2).to_u8().unwrap();
        let v = fdr_fx_raw(image).iter().map(f).collect();
        GrayImage::from_raw(image.width(), image.height(), v).unwrap()
    }

    pub fn fdr_fy(image: &GrayImage) -> GrayImage {
        let f = |subpixel: &i16| ((subpixel + i16::from(u8::MAX)) / 2).to_u8().unwrap();
        let v = fdr_fy_raw(image).iter().map(f).collect();
        GrayImage::from_raw(image.width(), image.height(), v).unwrap()
    }

    pub fn fdr_grad(image: &GrayImage) -> GrayImage {
        let v: Vec<u8> = fdr_fx_raw(image)
            .iter()
            .zip(fdr_fy_raw(image).iter())
            .map(|(x, y)| (f32::from(*x), f32::from(*y)))
            .map(|(x, y)| 255.0 * x.hypot(y) / 255f32.hypot(255f32))
            .map(|subpixel| subpixel.to_u8().unwrap())
            .collect();
        GrayImage::from_raw(image.width(), image.height(), v).unwrap()
    }

    fn sdr_fxx_raw(image: &GrayImage) -> LumaImage<i16> {
        image.filter(Kernel::new(vec![1, -2, 1], 1, 0))
    }

    fn sdr_fyy_raw(image: &GrayImage) -> LumaImage<i16> {
        image.filter(Kernel::new(vec![1, -2, 1], 0, 1))
    }

    pub fn sdr_fxx(image: &GrayImage, sigma: f32) -> GrayImage {
        let noise_reduced = gaussian(image, FilterView::new(1, 1), sigma);
        let f = |subpixel: &i16| ((subpixel + 2 * i16::from(u8::MAX)) / 4).to_u8().unwrap();
        let v = sdr_fxx_raw(&noise_reduced).iter().map(f).collect();
        GrayImage::from_vec(image.width(), image.height(), v).unwrap()
    }

    pub fn sdr_fyy(image: &GrayImage, sigma: f32) -> GrayImage {
        let noise_reduced = gaussian(image, FilterView::new(1, 1), sigma);
        let f = |subpixel: &i16| ((subpixel + 2 * i16::from(u8::MAX)) / 4).to_u8().unwrap();
        let v = sdr_fyy_raw(&noise_reduced).iter().map(f).collect();
        GrayImage::from_vec(image.width(), image.height(), v).unwrap()
    }

    pub fn laplacian(image: &GrayImage, sigma: f32, strength: f32) -> GrayImage {
        let noise_reduced = gaussian(image, FilterView::new(1, 1), sigma);
        let sdr_fxx = sdr_fxx_raw(&noise_reduced);
        let sdr_fyy = sdr_fyy_raw(&noise_reduced);

        let (width, height) = image.dimensions();
        let mut filtered = GrayImage::new(width, height);
        for (x, y, Luma([subpixel])) in filtered.enumerate_pixels_mut() {
            let mut acc = 0;
            for i in [-1, 1] {
                let view_x = add_wrap_within(x, i, width);
                acc += sdr_fxx[(view_x, y)][0];
                let view_y = add_wrap_within(y, i, height);
                acc += sdr_fyy[(x, view_y)][0];
            }
            *subpixel = (f32::from(*subpixel) - strength * f32::from(acc))
                .clamp(f32::from(u8::MIN), f32::from(u8::MAX))
                .to_u8()
                .unwrap();
        }
        filtered
    }

    fn sobel_horizontal_raw(image: &GrayImage) -> LumaImage<i16> {
        image.filter(Kernel::new(vec![-1, 0, 1, -2, 0, 2, -1, 0, 1], 1, 1))
    }

    fn sobel_vertical_raw(image: &GrayImage) -> LumaImage<i16> {
        image.filter(Kernel::new(vec![-1, -2, -1, 0, 0, 0, 1, 2, 1], 1, 1))
    }

    pub fn sobel_horizontal(image: &GrayImage) -> GrayImage {
        let f = |subpixel: &i16| ((subpixel + 4 * i16::from(u8::MAX)) / 8).to_u8().unwrap();
        let v = sobel_horizontal_raw(image).iter().map(f).collect();
        GrayImage::from_raw(image.width(), image.height(), v).unwrap()
    }

    pub fn sobel_vertical(image: &GrayImage) -> GrayImage {
        let f = |subpixel: &i16| ((subpixel + 4 * i16::from(u8::MAX)) / 8).to_u8().unwrap();
        let v = sobel_vertical_raw(image).iter().map(f).collect();
        GrayImage::from_raw(image.width(), image.height(), v).unwrap()
    }

    fn sobel_raw(image: &GrayImage) -> (LumaImage<f32>, LumaImage<f32>) {
        let (width, height) = image.dimensions();
        let grad_x = sobel_horizontal_raw(&image);
        let grad_y = sobel_vertical_raw(&image);
        let mut grad = LumaImage::<f32>::new(width, height);
        let mut phase = LumaImage::<f32>::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let (g_x, g_y) = (f32::from(grad_x[(x, y)][0]), f32::from(grad_y[(x, y)][0]));
                grad[(x, y)] = [g_x.hypot(g_y)].into();
                phase[(x, y)] = [g_y.atan2(g_x)].into();
            }
        }
        (grad, phase)
    }

    pub fn sobel(image: &GrayImage) -> (GrayImage, LumaImage<f32>) {
        let (width, height) = image.dimensions();
        let (grad, phase) = sobel_raw(image);
        let v = grad
            .iter()
            .map(|subpixel| {
                f32::from(u8::MAX) * subpixel
                    / (4.0 * f32::from(u8::MAX)).hypot(4.0 * f32::from(u8::MAX))
            })
            .map(|subpixel| subpixel.to_u8().unwrap())
            .collect();
        (GrayImage::from_raw(width, height, v).unwrap(), phase)
    }

    pub fn canny(image: &GrayImage, sigma: f32, (low_th, high_th): (f32, f32)) -> GrayImage {
        use std::f32::consts::{FRAC_PI_8, PI};

        let (width, height) = image.dimensions();
        let noise_reduced = gaussian(image, FilterView::new(1, 1), sigma);
        let (grad, phase) = sobel_raw(&noise_reduced);

        // Non maximum suppression
        let (mut grad, grad_og) = (grad.clone(), grad);
        for ((x, y, Luma([grad])), &phase) in grad.enumerate_pixels_mut().zip(phase.iter()) {
            let phase = if phase < 0.0 { phase } else { phase + PI };
            let adj_h = [(-1, 0), (1, 0)]; // horizontal
            let adj_p = [(-1, -1), (1, 1)]; // proportional
            let adj_v = [(0, -1), (0, 1)]; // vertical
            let adj_d = [(-1, 1), (1, -1)]; // disporoportional
            let adj = match phase {
                v if !((1.0 * FRAC_PI_8)..(7.0 * FRAC_PI_8)).contains(&v) => adj_v,
                v if ((1.0 * FRAC_PI_8)..(3.0 * FRAC_PI_8)).contains(&v) => adj_d,
                v if ((3.0 * FRAC_PI_8)..(5.0 * FRAC_PI_8)).contains(&v) => adj_h,
                v if ((5.0 * FRAC_PI_8)..(7.0 * FRAC_PI_8)).contains(&v) => adj_p,
                _ => unreachable!(),
            };
            for (l, k) in adj {
                let (view_x, view_y) =
                    (add_wrap_within(x, l, width), add_wrap_within(y, k, height));
                let adj_subpixel = grad_og[(view_x, view_y)][0];
                if adj_subpixel > *grad {
                    *grad = 0.0;
                    break;
                }
            }
        }

        // Hysterisis Threshold Process
        let mut edges = Vec::with_capacity((width * height / 2) as usize);
        let (mut filtered, grad) = (GrayImage::new(width, height), grad);
        for y in 0..filtered.height() {
            for x in 0..filtered.width() {
                if grad[(x, y)][0] >= high_th {
                    filtered[(x, y)] = [u8::MAX].into();
                    edges.push((x, y));
                    while !edges.is_empty() {
                        let adj = [
                            (-1, -1),
                            (0, -1),
                            (1, -1),
                            (-1, 0),
                            (1, 0),
                            (-1, 1),
                            (0, 1),
                            (1, 1),
                        ];
                        let (adj_x, adj_y) = edges.pop().unwrap();
                        for (l, k) in adj {
                            let (adj_x, adj_y) = (
                                add_wrap_within(adj_x, l, width),
                                add_wrap_within(adj_y, k, height),
                            );
                            let subpixel = &mut filtered[(adj_x, adj_y)][0];
                            if grad[(adj_x, adj_y)][0] >= low_th && *subpixel == 0 {
                                *subpixel = u8::MAX;
                                edges.push((adj_x, adj_y));
                            }
                        }
                    }
                }
            }
        }
        filtered
    }

    fn add_wrap_within(v: u32, v_idx: i32, bound: u32) -> u32 {
        let v = i64::from(v) + i64::from(v_idx);
        let bound = i64::from(bound);
        let wrapped = match v {
            v if v < 0 => v + bound,
            v if bound <= v => v - bound,
            v => v,
        };
        wrapped.to_u32().unwrap()
    }
}

enum Command {
    FirstDeriv,
    Laplacian,
    Sobel,
    Canny,
}

impl TryFrom<String> for Command {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "fdr" => Ok(Command::FirstDeriv),
            "lap" => Ok(Command::Laplacian),
            "sob" => Ok(Command::Sobel),
            "can" => Ok(Command::Canny),
            _ => Err("Command should be fdr, lap, sob, or can".to_string()),
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
    let mut args = std::env::args().skip(1).take(2);
    let command: Command = args.next().expect("No arguments provided").try_into()?;
    let input_path: PathBuf = args.next().expect("No file argument provided").into();
    let img = image::open(input_path.as_path())?.into_luma8();

    macro_rules! apply_filter {
        ($filtered:expr, $postfix:expr) => {
            $filtered.save(postfix_path(input_path.clone(), $postfix))
        };
        ($filtered:expr, $format:expr, $($args:expr),*) => {
            $filtered.save(postfix_path(input_path.clone(), &format!($format, $($args),*)))
        }
    }

    match command {
        Command::FirstDeriv => {
            apply_filter!(filter::fdr_fx(&img), "_fdr_x")?;
            apply_filter!(filter::fdr_fy(&img), "_fdr_y")?;
            apply_filter!(filter::fdr_grad(&img), "_fdr_m")?;
        }
        Command::Laplacian => {
            for sigma in [1.0, 2.0, 4.0] {
                println!("Filtering sigma = {}", sigma);
                apply_filter!(
                    filter::gaussian(&img, filter::FilterView::new(1, 1), sigma),
                    "_gauss_{}",
                    sigma
                )?;
                apply_filter!(filter::sdr_fxx(&img, sigma), "_sdr_x_{}", sigma)?;
                apply_filter!(filter::sdr_fyy(&img, sigma), "_sdr_y_{}", sigma)?;
                apply_filter!(filter::laplacian(&img, sigma, 1.0), "_lap_{}", sigma)?;
            }
        }
        Command::Sobel => {
            apply_filter!(filter::sobel_horizontal(&img), "_sob_x")?;
            apply_filter!(filter::sobel_vertical(&img), "_sob_y")?;
            apply_filter!(filter::sobel(&img).0, "_sob_m")?;
        }
        Command::Canny => {
            for sigma in [1.0, 2.0, 4.0] {
                apply_filter!(filter::canny(&img, sigma, (120.0, 150.0)), "_can_{}", sigma)?;
            }
        }
    }
    Ok(())
}
