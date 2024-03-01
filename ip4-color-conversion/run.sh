#!/bin/sh

file=${1:-images/lenna_color.bmp}

cargo build --release
./target/release/ip4-color-conversion rgb2hsv "$file"
./target/release/ip4-color-conversion rgb2gray "$file"
