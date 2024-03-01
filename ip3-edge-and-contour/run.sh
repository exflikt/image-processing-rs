#!/bin/sh

file=${1:-circle/small-circle.bmp}

cargo build --release
./target/release/ip3-edge-and-contour fdr "$file"
./target/release/ip3-edge-and-contour lap "$file"
./target/release/ip3-edge-and-contour sob "$file"
./target/release/ip3-edge-and-contour can "$file"
