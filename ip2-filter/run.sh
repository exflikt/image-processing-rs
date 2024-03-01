#!/bin/sh

cargo build --release
./target/release/ip2-filter box images/lenna_gray.bmp
./target/release/ip2-filter gauss images/lenna_gray.bmp
./target/release/ip2-filter min images/lenna_gray_noise.bmp
./target/release/ip2-filter max images/lenna_gray_noise.bmp
./target/release/ip2-filter med images/lenna_gray_noise.bmp

#./target/release/ip2-filter box dummy/lenna_color.bmp
#./target/release/ip2-filter gauss dummy/lenna_color.bmp
#./target/release/ip2-filter min dummy/lenna_color.bmp
#./target/release/ip2-filter max dummy/lenna_color.bmp
#./target/release/ip2-filter med dummy/lenna_color.bmp
