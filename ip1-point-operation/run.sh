#!/bin/sh

cargo b -r
./target/release/ip1-point-operation inv images/lenna_gray.bmp
./target/release/ip1-point-operation bin images/lenna_gray.bmp
./target/release/ip1-point-operation eq images/lenna_gray.bmp
./target/release/ip1-point-operation gc images/lenna_gray.bmp
