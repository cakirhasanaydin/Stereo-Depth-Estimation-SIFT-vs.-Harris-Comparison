# Stereo Depth Estimation: SIFT vs. Harris Comparison

This repository implements a comprehensive computer vision project comparing **SIFT (Scale-Invariant Feature Transform)** and **Harris Corner Detection** pipelines for stereo vision and depth (disparity) estimation. Additionally, it features an OpenCV SGBM benchmark and various robustness evaluations against noise, scale, and rotation changes.

---

## 📌 Project Overview & Rationale
Stereo depth estimation relies on finding point correspondences between left and right stereo image pairs to calculate disparity and reconstruct depth maps. 
* **SIFT:** Utilizes scale- and rotation-invariant keypoint detection along with 128-dimensional robust descriptors.
* **Harris:** Employs gradient-based corner detection with localized patch-based matching.
* **Why not Canny?** As explicitly noted in the project design, Canny edge detection is excluded from feature-based stereo matching because it only outputs edge maps without generating descriptors or enabling point-to-point correspondences.

---

## 🚀 Features
* **Dual Feature Pipelines:** Independent implementation and benchmarking of SIFT and Harris matching algorithms.
* **OpenCV SGBM Reference:** Includes a semi-global block matching (SGBM) reference implementation for quality comparison.
* **Robustness & Stress Tests:**
  * **Noise Test:** Evaluates performance degradation under varying Gaussian noise levels ($\sigma$).
  * **Scale Test:** Assesses resilience to resolution/scale changes in the right image.
  * **Rotation Test:** Analyzes stability under angular rotation variants.
* **Automated Visualizations:** Generates and saves analytical graphs and a summary comparison table (`sonuclar/` klasörüne otomatik çıktı verir).

---

## 🛠️ Requirements & Tech Stack
* **Python** (Compatible with `py` or `python`)
* **OpenCV** (`cv2`)
* **NumPy**
* **Matplotlib**

---

## 📦 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/cakirhasanaydin/Stereo-Depth-Estimation-SIFT-vs.-Harris-Comparison.git](https://github.com/cakirhasanaydin/Stereo-Depth-Estimation-SIFT-vs.-Harris-Comparison.git)
   cd Stereo-Depth-Estimation-SIFT-vs.-Harris-Comparison

2. Install dependencies:

Bash
pip install opencv-python numpy matplotlib

Run the script:

Default Mode (Looks for left.jpg and right.jpg in the root directory):

Bash
py stereo_depth.py

Custom Image Paths:

Bash
py stereo_depth.py --left foto1.jpg --right foto2.jpg

Demo Mode (Generates synthetic stereo pairs and runs all tests automatically):

Bash
py stereo_depth.py --demo


Metric / Property,SIFT Pipeline,Harris Pipeline
Feature / Corner Type,Scale-invariant keypoints,Intensity-based corners
Descriptor,128-D feature vector,None (Patch matching required)
Matching Method,BFMatcher + Lowe's Ratio Test,SSD (Sum of Squared Differences) patch window
Scale Invariance,High,Low
Rotation Invariance,High,Partial
Depth Estimation Quality,Highly accurate disparity maps,Moderate / Sensitive to patch shifts


Output Visualizations

When executed, the script automatically produces the following diagnostic plots in the output folder:

1_feature_detection.png — Keypoints and corner distributions.

2_matching_quality.png — Inlier/outlier match visualizations via RANSAC.

3_disparity_maps.png — SIFT vs. Harris vs. OpenCV SGBM depth disparity maps.

4_performance.png — Match counts, inlier ratios, and execution times.

5_noise_test.png — Noise robustness curves.

6_scale_test.png — Scale variation impact.

7_rotation_test.png — Rotation angle resilience.

8_summary_table.png — Comprehensive text-to-table metric summary.


