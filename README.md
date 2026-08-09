Stereo Depth Estimation: SIFT vs. Harris Comparison

This project is a computer vision application that aims to generate a depth (disparity) map using stereo image pairs taken from two different cameras. The project strictly focuses on traditional image processing and feature extraction algorithms; **no deep learning** techniques are used.

**SIFT** (Scale-Invariant Feature Transform) and **Harris Corner Detector** algorithms are utilized to find point matches across stereo images and extract sparse disparity maps. The results are then compared with the **OpenCV SGBM** (Semi-Global Block Matching) algorithm, which serves as a baseline reference.

 Features

- **Feature Extraction & Matching:** Keypoint and corner detection using SIFT and Harris algorithms.
- **Sparse Disparity Maps:** Calculation of pixel shift amounts based on matched feature points.
- **Robustness Tests:** 
  - *Gaussian Noise Test*
  - *Scale Variation Test*
  - *Rotation Test*
- **Visualization & Reporting:** Automated generation of high-quality plots for feature detection, matching quality, disparity maps, and performance metrics using `matplotlib`.
- **Demo Mode:** Built-in capability to generate and test a synthetic stereo image pair without requiring external image files.

