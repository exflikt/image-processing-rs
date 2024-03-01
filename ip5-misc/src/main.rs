use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Image Processing Module
pub mod ip {
    use image::{GrayImage, ImageBuffer, Luma, Primitive, RgbImage};
    use num_traits::ToPrimitive;

    pub fn template_match(img: GrayImage, tmp: GrayImage) -> ImageBuffer<Luma<usize>, Vec<usize>> {
        let (width, height) = img.dimensions();
        let (tmp_width, tmp_height) = tmp.dimensions();
        assert!(width >= tmp_width && height >= tmp_height);

        let mut res =
            ImageBuffer::<Luma<usize>, Vec<usize>>::new(width - tmp_width, height - tmp_height);
        for (x, y, ssd) in res.enumerate_pixels_mut() {
            for tmp_y in 0..tmp_height {
                for tmp_x in 0..tmp_width {
                    let (pixel, tmp) = (img[(x + tmp_x, y + tmp_y)], tmp[(tmp_x, tmp_y)]);
                    ssd[0] += usize::from(pixel[0]).abs_diff(usize::from(tmp[0])).pow(2);
                }
            }
        }
        res
    }

    trait BinaryExt<T: Primitive> {
        fn binarize(&self, thresh: T) -> Self;
    }

    impl<T: Primitive> BinaryExt<T> for ImageBuffer<Luma<T>, Vec<T>> {
        fn binarize(&self, thresh: T) -> Self {
            let (width, height) = self.dimensions();
            Self::from_raw(
                width,
                height,
                self.iter()
                    .map(|&int| {
                        (int <= thresh)
                            .then_some(<T as Primitive>::DEFAULT_MIN_VALUE)
                            .unwrap_or(<T as Primitive>::DEFAULT_MAX_VALUE)
                    })
                    .collect(),
            )
            .unwrap()
        }
    }

    pub fn hough(
        img: GrayImage,
        theta_steps: u32,
        rho_steps: u32,
        voting_thresh: usize,
    ) -> GrayImage {
        struct PolarSpace {
            theta_range: std::ops::Range<f32>,
            theta_steps: u32,
            rho_max: f32,
            rho_steps: u32,
        }

        impl PolarSpace {
            fn theta(&self, theta_step: u32) -> f32 {
                assert!(theta_step <= self.theta_steps);
                let std::ops::Range { start, end } = self.theta_range;
                (end - start) * theta_step.to_f32().unwrap() / self.theta_steps.to_f32().unwrap()
            }

            fn rho(&self, rho_step: u32) -> f32 {
                assert!(rho_step <= self.rho_steps);
                -self.rho_max
                    + (2. * self.rho_max) * rho_step.to_f32().unwrap()
                        / self.rho_steps.to_f32().unwrap()
            }

            fn at(&self, theta_step: u32, rho_step: u32) -> [f32; 2] {
                [self.theta(theta_step), self.rho(rho_step)]
            }

            fn rho_step_from_euclidian(&self, theta_step: u32, [x, y]: [u32; 2]) -> u32 {
                let theta = self.theta(theta_step);
                let [x, y] = [x, y].map(|v| v.to_f32().unwrap());
                let rho = x * theta.cos() + y * theta.sin();
                let rho_steps = self.rho_steps.to_f32().unwrap();
                (rho_steps * (rho + self.rho_max) / (2. * self.rho_max))
                    .to_u32()
                    .unwrap()
            }
        }

        let (width, height) = img.dimensions();
        let [width_f, height_f] = [width, height].map(|v| v.to_f32().unwrap());
        let polar_space = PolarSpace {
            theta_range: 0.0..std::f32::consts::PI,
            theta_steps,
            rho_max: width_f.hypot(height_f),
            rho_steps,
        };

        let binarized = img.binarize(u8::MAX / 2);
        let mut voting: std::collections::HashMap<[u32; 2], usize> = Default::default();
        for theta_step in 0..polar_space.theta_steps {
            for (x, y, pixel) in binarized.enumerate_pixels() {
                if pixel[0] == u8::MAX {
                    let rho_step = polar_space.rho_step_from_euclidian(theta_step, [x, y]);
                    voting
                        .entry([theta_step, rho_step])
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                }
            }
        }

        let mut transformed = GrayImage::new(width, height);
        for [theta, rho] in voting
            .into_iter()
            .filter(|&(_polar_coord_steps, v)| v >= voting_thresh)
            .map(|([theta_step, rho_step], _)| polar_space.at(theta_step, rho_step))
        {
            for x in 0..width {
                let x_f = x.to_f32().unwrap();
                // x cos(θ) + y sin(θ) = ρ
                let y_f = -x_f * theta.cos() / theta.sin() + (rho / theta.sin());
                if 0. <= y_f && y_f <= height.to_f32().unwrap() {
                    transformed[(x, y_f.to_u32().unwrap())] = [u8::MAX].into();
                }
            }
            for y in 0..height {
                let y_f = y.to_f32().unwrap();
                let x_f = -y_f * theta.sin() / theta.cos() + (rho / theta.cos());
                if 0. <= x_f && x_f <= width.to_f32().unwrap() {
                    transformed[(x_f.to_u32().unwrap(), y)] = [u8::MAX].into();
                }
            }
        }
        transformed
    }

    pub struct Affine {
        rot_mat: [[f32; 2]; 2],
        tr_vec: [f32; 2],
    }

    impl Affine {
        pub fn new(rot_mat: [[f32; 2]; 2], tr_vec: [f32; 2]) -> Self {
            Self { rot_mat, tr_vec }
        }

        pub fn rotate(&self, [x, y]: [f32; 2]) -> [f32; 2] {
            let rot = self.rot_mat;
            [rot[0][0] * x + rot[0][1] * y, rot[1][0] * x + rot[1][1] * y]
        }

        pub fn translate(&self, [x, y]: [f32; 2]) -> [f32; 2] {
            let tr = self.tr_vec;
            [x + tr[0], y + tr[1]]
        }

        pub fn transform(&self, pos: [f32; 2]) -> [f32; 2] {
            // / x' \   / a b \ / x \   / e \
            // \ y' / = \ c d / \ y / + \ f /
            self.translate(self.rotate(pos))
        }

        pub fn inv_transform(&self, [x, y]: [f32; 2]) -> [f32; 2] {
            // / x' - e \   / a b \ / x \
            // \ y' - f / = \ c d / \ y /
            //              /  d -b \ / x' - e \   / x \
            // (ad - bc)^-1 \ -c  a / \ y' - f / = \ y /
            let (rot, tr) = (self.rot_mat, self.tr_vec);
            let coef = 1. / (rot[0][0] * rot[1][1] - rot[1][0] * rot[0][1]);
            [
                coef * (rot[1][1] * (x - tr[0]) - rot[0][1] * (y - tr[1])),
                coef * (-rot[1][0] * (x - tr[0]) + rot[0][0] * (y - tr[1])),
            ]
        }
    }

    pub fn affine(img: RgbImage, affine: Affine) -> RgbImage {
        let (width, height) = img.dimensions();
        let (width, height) = (width.to_f32().unwrap(), height.to_f32().unwrap());
        let corners = [[0., 0.], [width, 0.], [0., height], [width, height]];
        let rotated_corners = corners.map(|pos| affine.rotate(pos));
        let rotated_corners_x = rotated_corners.map(|[x, _]| x);
        let rotated_corners_y = rotated_corners.map(|[_, y]| y);
        let min_x = rotated_corners_x.into_iter().reduce(f32::min).unwrap();
        let min_y = rotated_corners_y.into_iter().reduce(f32::min).unwrap();
        let max_x = rotated_corners_x.into_iter().reduce(f32::max).unwrap();
        let max_y = rotated_corners_y.into_iter().reduce(f32::max).unwrap();
        let (new_width, new_height) = (max_x - min_x, max_y - min_y);
        let [new_width, new_height] = [new_width, new_height].map(|v| v.to_u32().unwrap());

        let mut transformed = RgbImage::new(new_width, new_height);
        for (x, y, pixel) in transformed.enumerate_pixels_mut() {
            let [x, y] = [x.to_f32().unwrap() + min_x, y.to_f32().unwrap() + min_y];
            let [org_x, org_y] = affine.inv_transform([x, y]);
            if 0. <= org_x && org_x < width && 0. <= org_y && org_y < height {
                let [org_x, org_y] = [org_x, org_y].map(|v| v.to_u32().unwrap());
                *pixel = img[(org_x, org_y)];
            }
        }
        transformed
    }
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    TemplateMatching {
        img_path: PathBuf,
        template_path: PathBuf,
    },
    Hough {
        img_path: PathBuf,
        theta_steps: u32,
        rho_steps: u32,
        voting_thresh: usize,
    },
    Affine {
        img_path: PathBuf,
    },
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
    match Cli::parse().command {
        Command::TemplateMatching {
            img_path,
            template_path,
        } => {
            let img = image::open(img_path.as_path())?.into_luma8();
            let tmp = image::open(template_path.as_path())?.into_luma8();
            let (tmp_width, tmp_height) = tmp.dimensions();
            let ssd = ip::template_match(img, tmp);
            for (rel_x, rel_y, ssd) in ssd.enumerate_pixels() {
                let [x, y] = [rel_x + tmp_width / 2, rel_y + tmp_height / 2];
                println!("{},{},{}", x, y, ssd[0]);
            }
        }
        Command::Hough {
            img_path,
            theta_steps,
            rho_steps,
            voting_thresh,
        } => {
            let _img = image::open(img_path.as_path())?.into_luma8();
            let postfix = format!("_hough_{theta_steps}_{rho_steps}_{voting_thresh}");
            ip::hough(_img, theta_steps, rho_steps, voting_thresh)
                .save(postfix_path(img_path, &postfix))?;
        }
        Command::Affine { img_path } => {
            let img = image::open(img_path.as_path())?.into_rgb8();

            macro_rules! affine {
                ($desc:expr, $rot:expr, $tr:expr) => {
                    let path = postfix_path(img_path.clone(), &format!("_affine_{}", $desc));
                    println!("Processing {:?} ...", path);
                    ip::affine(img.clone(), ip::Affine::new($rot, $tr)).save(path)?;
                };
            }

            affine!("scale_2", [[2., 0.], [0., 2.]], [0., 0.]);
            affine!("scale_2x", [[2., 0.], [0., 1.]], [0., 0.]);
            affine!("scale_2y", [[1., 0.], [0., 2.]], [0., 0.]);
            affine!("tr_100,100", [[1., 0.], [0., 1.]], [100., 100.]);
            let theta = std::f32::consts::FRAC_PI_4;
            affine!(
                "rot_pi_4",
                [[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]],
                [0., 0.]
            );
            let theta = -5. * std::f32::consts::FRAC_PI_6;
            affine!(
                "rot_neg_5pi_6",
                [[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]],
                [0., 0.]
            );
            let theta = std::f32::consts::FRAC_PI_6;
            affine!("sheer_x_pi_6", [[1., theta.tan()], [0., 1.]], [0., 0.]);
            affine!("sheer_x_neg_pi_6", [[1., -theta.tan()], [0., 1.]], [0., 0.]);
            affine!("sheer_y_pi_6", [[1., 0.], [theta.tan(), 1.]], [0., 0.]);
            affine!("sheer_y_neg_pi_6", [[1., 0.], [-theta.tan(), 1.]], [0., 0.]);
        }
    }
    Ok(())
}
