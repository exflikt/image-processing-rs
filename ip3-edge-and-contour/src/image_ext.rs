//! Convinient traits for `image::ImageBuffer`, mainly provide subpixel operations.

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
    fn get_subpixel(&self, x: u32, y: u32, ch: u8) -> <P as Pixel>::Subpixel;

    fn enumerate_subpixels(&self) -> EnumerateSubpixels<P>;
}

pub trait ImageBufferExtMut<P: Pixel> {
    fn get_subpixel_mut(&mut self, x: u32, y: u32, ch: u8) -> &mut <P as Pixel>::Subpixel;

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
