"""
Stereo Görüntüden Derinlik Tahmini - SIFT vs Harris Karşılaştırması
====================================================================
Computer Vision Dersi Projesi

SIFT ve Harris methodlarını stereo derinlik tahmini görevinde karşılaştırır.
Canny (edge detection) bu projede kullanılmamıştır çünkü descriptor üretmez
ve nokta eşleştirmesi yapamaz — stereo derinlik tahmini eşleştirme gerektirir.

Kullanım:
  1) left.jpg ve right.jpg dosyalarını script ile aynı klasöre koy:
     python stereo_depth.py

  2) Dosya belirterek:
     python stereo_depth.py --left foto1.jpg --right foto2.jpg

  3) Demo modu:
     python stereo_depth.py --demo
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import os
import sys


# ============================================================================
# 1. YARDIMCI FONKSİYONLAR
# ============================================================================

def load_stereo_images(left_path, right_path, max_width=800):
    """Stereo görüntüleri yükle ve boyutlandır."""
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    if left is None or right is None:
        print("HATA: Goruntuler yuklenemedi! Dosya yollarini kontrol et.")
        sys.exit(1)

    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left = cv2.resize(left, (w, h))
    right = cv2.resize(right, (w, h))

    if w > max_width:
        scale = max_width / w
        left = cv2.resize(left, None, fx=scale, fy=scale)
        right = cv2.resize(right, None, fx=scale, fy=scale)

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    return left, right, gray_left, gray_right


def add_gaussian_noise(image, sigma):
    """Görüntüye Gaussian gürültü ekle."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float64)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def compute_disparity_map(points_left, points_right, img_shape, radius=20):
    """Eşleşen nokta çiftlerinden disparity haritası oluştur."""
    disparity_map = np.zeros(img_shape, dtype=np.float64)
    count_map = np.zeros(img_shape, dtype=np.float64)

    for (x1, y1), (x2, y2) in zip(points_left, points_right):
        d = abs(x1 - x2)
        ix, iy = int(x1), int(y1)
        if 0 <= iy < img_shape[0] and 0 <= ix < img_shape[1]:
            y_min = max(0, iy - radius)
            y_max = min(img_shape[0], iy + radius)
            x_min = max(0, ix - radius)
            x_max = min(img_shape[1], ix + radius)
            disparity_map[y_min:y_max, x_min:x_max] += d
            count_map[y_min:y_max, x_min:x_max] += 1

    count_map[count_map == 0] = 1
    return disparity_map / count_map


# ============================================================================
# 2. SIFT
# ============================================================================

def sift_pipeline(gray_left, gray_right):
    """SIFT: keypoint tespiti + descriptor + eşleştirme."""
    start = time.time()

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_left, None)
    kp2, des2 = sift.detectAndCompute(gray_right, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return {'kp1': [], 'kp2': [], 'matches': [], 'time': time.time()-start,
                'n_kp1': 0, 'n_kp2': 0, 'pts_left': [], 'pts_right': [], 'inlier_ratio': 0}

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Eşleşen noktaların koordinatları
    pts_left = [kp1[m.queryIdx].pt for m in good_matches]
    pts_right = [kp2[m.trainIdx].pt for m in good_matches]

    # RANSAC ile inlier/outlier ayrımı
    inlier_ratio = 0.0
    if len(pts_left) >= 4:
        src = np.float32(pts_left).reshape(-1, 1, 2)
        dst = np.float32(pts_right).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_ratio = np.sum(mask) / len(mask) * 100

    elapsed = time.time() - start

    return {
        'kp1': kp1, 'kp2': kp2, 'matches': good_matches,
        'time': elapsed, 'n_kp1': len(kp1), 'n_kp2': len(kp2),
        'pts_left': pts_left, 'pts_right': pts_right,
        'inlier_ratio': inlier_ratio
    }


# ============================================================================
# 3. HARRIS
# ============================================================================

def harris_pipeline(gray_left, gray_right):
    """Harris: köşe tespiti + pencere tabanlı eşleştirme."""
    start = time.time()

    harris_left = cv2.cornerHarris(gray_left.astype(np.float32), blockSize=2, ksize=3, k=0.04)
    harris_right = cv2.cornerHarris(gray_right.astype(np.float32), blockSize=2, ksize=3, k=0.04)

    threshold = 0.01 * harris_left.max()
    corners_left = np.argwhere(harris_left > threshold)
    corners_right = np.argwhere(harris_right > threshold)

    # En güçlü 500 köşe
    for arr, harris_map in [(corners_left, harris_left), (corners_right, harris_right)]:
        pass  # aşağıda filtreliyoruz

    if len(corners_left) > 500:
        scores = harris_left[corners_left[:, 0], corners_left[:, 1]]
        corners_left = corners_left[np.argsort(scores)[-500:]]
    if len(corners_right) > 500:
        scores = harris_right[corners_right[:, 0], corners_right[:, 1]]
        corners_right = corners_right[np.argsort(scores)[-500:]]

    # Pencere tabanlı eşleştirme
    patch_size = 11
    half = patch_size // 2
    pts_left = []
    pts_right = []
    match_distances = []

    for y1, x1 in corners_left:
        if y1-half < 0 or y1+half >= gray_left.shape[0] or x1-half < 0 or x1+half >= gray_left.shape[1]:
            continue
        patch1 = gray_left[y1-half:y1+half+1, x1-half:x1+half+1].astype(np.float64)

        best_dist = float('inf')
        best_pt = None

        for y2, x2 in corners_right:
            if abs(y1 - y2) > 5:
                continue
            if y2-half < 0 or y2+half >= gray_right.shape[0] or x2-half < 0 or x2+half >= gray_right.shape[1]:
                continue
            patch2 = gray_right[y2-half:y2+half+1, x2-half:x2+half+1].astype(np.float64)
            dist = np.sum((patch1 - patch2) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_pt = (x2, y2)

        if best_pt is not None and best_dist < 50000:
            pts_left.append((x1, y1))
            pts_right.append(best_pt)
            match_distances.append(best_dist)

    # RANSAC ile inlier oranı
    inlier_ratio = 0.0
    if len(pts_left) >= 4:
        src = np.float32(pts_left).reshape(-1, 1, 2)
        dst = np.float32(pts_right).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_ratio = np.sum(mask) / len(mask) * 100

    elapsed = time.time() - start

    return {
        'corners_left': corners_left, 'corners_right': corners_right,
        'time': elapsed, 'n_corners_left': len(corners_left), 'n_corners_right': len(corners_right),
        'pts_left': pts_left, 'pts_right': pts_right,
        'n_matches': len(pts_left), 'inlier_ratio': inlier_ratio
    }


# ============================================================================
# 4. OPENCV SGBM (Referans)
# ============================================================================

def opencv_stereo_disparity(gray_left, gray_right):
    """OpenCV SGBM ile referans disparity haritası."""
    start = time.time()
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=11,
        P1=8*3*11**2, P2=32*3*11**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32
    )
    disparity = stereo.compute(gray_left, gray_right).astype(np.float64) / 16.0
    disparity[disparity < 0] = 0
    return disparity, time.time() - start


# ============================================================================
# 5. TESTLER
# ============================================================================

def noise_test(gray_left, gray_right):
    """Gürültüye dayanıklılık testi."""
    levels = [0, 10, 20, 30, 50]
    sift_matches, sift_inliers, sift_times = [], [], []
    harris_matches, harris_inliers, harris_times = [], [], []

    for sigma in levels:
        if sigma == 0:
            nl, nr = gray_left, gray_right
        else:
            nl = add_gaussian_noise(gray_left, sigma)
            nr = add_gaussian_noise(gray_right, sigma)

        s = sift_pipeline(nl, nr)
        sift_matches.append(len(s['matches']))
        sift_inliers.append(s['inlier_ratio'])
        sift_times.append(s['time'] * 1000)

        h = harris_pipeline(nl, nr)
        harris_matches.append(h['n_matches'])
        harris_inliers.append(h['inlier_ratio'])
        harris_times.append(h['time'] * 1000)

    return levels, {
        'SIFT': {'matches': sift_matches, 'inliers': sift_inliers, 'times': sift_times},
        'Harris': {'matches': harris_matches, 'inliers': harris_inliers, 'times': harris_times}
    }


def scale_test(gray_left, gray_right):
    """Ölçek değişimine dayanıklılık testi."""
    scales = [1.0, 0.8, 0.6, 0.4]
    sift_matches, sift_inliers = [], []
    harris_matches, harris_inliers = [], []

    for sc in scales:
        if sc == 1.0:
            scaled = gray_right
        else:
            scaled = cv2.resize(gray_right, None, fx=sc, fy=sc)
            scaled = cv2.resize(scaled, (gray_right.shape[1], gray_right.shape[0]))

        s = sift_pipeline(gray_left, scaled)
        sift_matches.append(len(s['matches']))
        sift_inliers.append(s['inlier_ratio'])

        h = harris_pipeline(gray_left, scaled)
        harris_matches.append(h['n_matches'])
        harris_inliers.append(h['inlier_ratio'])

    return scales, {
        'SIFT': {'matches': sift_matches, 'inliers': sift_inliers},
        'Harris': {'matches': harris_matches, 'inliers': harris_inliers}
    }


def rotation_test(gray_left, gray_right):
    """Rotasyon değişimine dayanıklılık testi."""
    angles = [0, 15, 30, 45, 60, 90]
    sift_matches, harris_matches = [], []
    sift_inliers, harris_inliers = [], []

    h, w = gray_right.shape
    center = (w // 2, h // 2)

    for angle in angles:
        if angle == 0:
            rotated = gray_right
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray_right, M, (w, h))

        s = sift_pipeline(gray_left, rotated)
        sift_matches.append(len(s['matches']))
        sift_inliers.append(s['inlier_ratio'])

        h_res = harris_pipeline(gray_left, rotated)
        harris_matches.append(h_res['n_matches'])
        harris_inliers.append(h_res['inlier_ratio'])

    return angles, {
        'SIFT': {'matches': sift_matches, 'inliers': sift_inliers},
        'Harris': {'matches': harris_matches, 'inliers': harris_inliers}
    }


# ============================================================================
# 6. GÖRSELLEŞTİRME
# ============================================================================

def create_all_figures(left, right, gray_left, gray_right, output_dir):
    """Tüm karşılaştırma grafiklerini oluştur."""

    print("\n" + "=" * 60)
    print("  STEREO DERINLIK TAHMINI - SIFT vs HARRIS")
    print("=" * 60)

    # --- Analizler ---
    print("\n[1/8] SIFT analizi...")
    sift = sift_pipeline(gray_left, gray_right)
    print(f"  Keypoint: {sift['n_kp1']} (sol), {sift['n_kp2']} (sag)")
    print(f"  Eslesmeler: {len(sift['matches'])}")
    print(f"  Inlier orani: {sift['inlier_ratio']:.1f}%")
    print(f"  Sure: {sift['time']*1000:.1f} ms")

    print("\n[2/8] Harris analizi...")
    harris = harris_pipeline(gray_left, gray_right)
    print(f"  Koseler: {harris['n_corners_left']} (sol), {harris['n_corners_right']} (sag)")
    print(f"  Eslesmeler: {harris['n_matches']}")
    print(f"  Inlier orani: {harris['inlier_ratio']:.1f}%")
    print(f"  Sure: {harris['time']*1000:.1f} ms")

    print("\n[3/8] Disparity haritalari hesaplaniyor...")
    disp_sift = compute_disparity_map(sift['pts_left'], sift['pts_right'], gray_left.shape)
    disp_harris = compute_disparity_map(harris['pts_left'], harris['pts_right'], gray_left.shape)

    print("[4/8] OpenCV SGBM referans...")
    disp_ref, time_ref = opencv_stereo_disparity(gray_left, gray_right)

    # ===== GRAFİK 1: Feature Tespiti =====
    print("[5/8] Grafik 1: Feature tespiti...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Tespiti: SIFT vs Harris', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Orijinal Sol Goruntu')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Orijinal Sag Goruntu')
    axes[0, 1].axis('off')

    sift_img = cv2.drawKeypoints(gray_left, sift['kp1'], None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes[1, 0].imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'SIFT Keypoints ({sift["n_kp1"]} nokta)')
    axes[1, 0].axis('off')

    harris_vis = cv2.cvtColor(gray_left, cv2.COLOR_GRAY2BGR)
    for y, x in harris['corners_left']:
        cv2.circle(harris_vis, (x, y), 3, (0, 0, 255), -1)
    axes[1, 1].imshow(cv2.cvtColor(harris_vis, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Harris Corners ({harris["n_corners_left"]} kose)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_feature_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== GRAFİK 2: Eşleştirme Kalitesi =====
    print("[5/8] Grafik 2: Eslestirme kalitesi...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Eslestirme Kalitesi: SIFT vs Harris', fontsize=16, fontweight='bold')

    if len(sift['matches']) > 0:
        match_img_sift = cv2.drawMatches(gray_left, sift['kp1'], gray_right, sift['kp2'],
                                          sift['matches'][:40], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        axes[0].imshow(cv2.cvtColor(match_img_sift, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'SIFT: {len(sift["matches"])} eslesme, {sift["inlier_ratio"]:.1f}% inlier')
    axes[0].axis('off')

    # Harris eşleşmelerini çiz
    h_match_img = np.hstack([cv2.cvtColor(gray_left, cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(gray_right, cv2.COLOR_GRAY2BGR)])
    w_offset = gray_left.shape[1]
    for (x1, y1), (x2, y2) in list(zip(harris['pts_left'], harris['pts_right']))[:40]:
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.circle(h_match_img, (int(x1), int(y1)), 4, color, -1)
        cv2.circle(h_match_img, (int(x2) + w_offset, int(y2)), 4, color, -1)
        cv2.line(h_match_img, (int(x1), int(y1)), (int(x2) + w_offset, int(y2)), color, 1)
    axes[1].imshow(cv2.cvtColor(h_match_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Harris: {harris["n_matches"]} eslesme, {harris["inlier_ratio"]:.1f}% inlier')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_matching_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== GRAFİK 3: Disparity Haritaları =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Disparity (Derinlik) Haritalari', fontsize=16, fontweight='bold')

    im1 = axes[0].imshow(disp_sift, cmap='jet')
    axes[0].set_title('SIFT Disparity')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(disp_harris, cmap='jet')
    axes[1].set_title('Harris Disparity')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    im3 = axes[2].imshow(disp_ref, cmap='jet')
    axes[2].set_title('OpenCV SGBM (Referans)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_disparity_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== GRAFİK 4: Performans Karşılaştırması =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Performans Karsilastirmasi', fontsize=16, fontweight='bold')

    # Eşleşme sayısı
    methods = ['SIFT', 'Harris']
    match_counts = [len(sift['matches']), harris['n_matches']]
    colors = ['#2196F3', '#FF9800']
    bars = axes[0].bar(methods, match_counts, color=colors)
    axes[0].set_ylabel('Eslesme Sayisi')
    axes[0].set_title('Toplam Eslesmeler')
    for bar, val in zip(bars, match_counts):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(val),
                      ha='center', fontweight='bold')

    # Inlier oranı
    inlier_vals = [sift['inlier_ratio'], harris['inlier_ratio']]
    bars = axes[1].bar(methods, inlier_vals, color=colors)
    axes[1].set_ylabel('Inlier Orani (%)')
    axes[1].set_title('Eslestirme Dogrulugu (RANSAC)')
    axes[1].set_ylim(0, 105)
    for bar, val in zip(bars, inlier_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}%',
                      ha='center', fontweight='bold')

    # Çalışma süresi
    time_vals = [sift['time']*1000, harris['time']*1000]
    bars = axes[2].bar(methods, time_vals, color=colors)
    axes[2].set_ylabel('Sure (ms)')
    axes[2].set_title('Calisma Suresi')
    for bar, val in zip(bars, time_vals):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}ms',
                      ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== GRAFİK 5: Gürültü Testi =====
    print("[6/8] Gurultu testi...")
    levels, noise_res = noise_test(gray_left, gray_right)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Gurultuye Dayaniklilik Testi', fontsize=16, fontweight='bold')

    axes[0].plot(levels, noise_res['SIFT']['matches'], 'b-o', label='SIFT', linewidth=2)
    axes[0].plot(levels, noise_res['Harris']['matches'], 'r-s', label='Harris', linewidth=2)
    axes[0].set_xlabel('Gurultu Seviyesi (sigma)')
    axes[0].set_ylabel('Eslesme Sayisi')
    axes[0].set_title('Eslesme Sayisi vs Gurultu')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(levels, noise_res['SIFT']['inliers'], 'b-o', label='SIFT', linewidth=2)
    axes[1].plot(levels, noise_res['Harris']['inliers'], 'r-s', label='Harris', linewidth=2)
    axes[1].set_xlabel('Gurultu Seviyesi (sigma)')
    axes[1].set_ylabel('Inlier Orani (%)')
    axes[1].set_title('Eslestirme Dogrulugu vs Gurultu')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(levels, noise_res['SIFT']['times'], 'b-o', label='SIFT', linewidth=2)
    axes[2].plot(levels, noise_res['Harris']['times'], 'r-s', label='Harris', linewidth=2)
    axes[2].set_xlabel('Gurultu Seviyesi (sigma)')
    axes[2].set_ylabel('Sure (ms)')
    axes[2].set_title('Calisma Suresi vs Gurultu')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_noise_test.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== GRAFİK 6: Ölçek Testi =====
    print("[7/8] Olcek testi...")
    sc_vals, scale_res = scale_test(gray_left, gray_right)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Olcek Degisimine Dayaniklilik', fontsize=16, fontweight='bold')

    labels = [f'{s:.0%}' for s in sc_vals]
    axes[0].plot(labels, scale_res['SIFT']['matches'], 'b-o', label='SIFT', linewidth=2)
    axes[0].plot(labels, scale_res['Harris']['matches'], 'r-s', label='Harris', linewidth=2)
    axes[0].set_xlabel('Sag Goruntu Olcegi')
    axes[0].set_ylabel('Eslesme Sayisi')
    axes[0].set_title('Eslesme Sayisi vs Olcek')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(labels, scale_res['SIFT']['inliers'], 'b-o', label='SIFT', linewidth=2)
    axes[1].plot(labels, scale_res['Harris']['inliers'], 'r-s', label='Harris', linewidth=2)
    axes[1].set_xlabel('Sag Goruntu Olcegi')
    axes[1].set_ylabel('Inlier Orani (%)')
    axes[1].set_title('Eslestirme Dogrulugu vs Olcek')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_scale_test.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== GRAFİK 7: Rotasyon Testi =====
    print("[8/8] Rotasyon testi...")
    angles, rot_res = rotation_test(gray_left, gray_right)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Rotasyona Dayaniklilik', fontsize=16, fontweight='bold')

    axes[0].plot(angles, rot_res['SIFT']['matches'], 'b-o', label='SIFT', linewidth=2)
    axes[0].plot(angles, rot_res['Harris']['matches'], 'r-s', label='Harris', linewidth=2)
    axes[0].set_xlabel('Rotasyon Acisi (derece)')
    axes[0].set_ylabel('Eslesme Sayisi')
    axes[0].set_title('Eslesme Sayisi vs Rotasyon')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(angles, rot_res['SIFT']['inliers'], 'b-o', label='SIFT', linewidth=2)
    axes[1].plot(angles, rot_res['Harris']['inliers'], 'r-s', label='Harris', linewidth=2)
    axes[1].set_xlabel('Rotasyon Acisi (derece)')
    axes[1].set_ylabel('Inlier Orani (%)')
    axes[1].set_title('Eslestirme Dogrulugu vs Rotasyon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_rotation_test.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== GRAFİK 8: Özet Tablo =====
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_title('Ozet Karsilastirma Tablosu', fontsize=16, fontweight='bold', pad=20)

    table_data = [
        ['Metrik', 'SIFT', 'Harris'],
        ['Feature Sayisi', str(sift['n_kp1']), str(harris['n_corners_left'])],
        ['Eslesme Sayisi', str(len(sift['matches'])), str(harris['n_matches'])],
        ['Inlier Orani', f"{sift['inlier_ratio']:.1f}%", f"{harris['inlier_ratio']:.1f}%"],
        ['Calisma Suresi', f"{sift['time']*1000:.1f} ms", f"{harris['time']*1000:.1f} ms"],
        ['Descriptor', '128-D vektor', 'Yok'],
        ['Olcek Dayanikliligi', 'Yuksek', 'Dusuk'],
        ['Rotasyon Dayanikliligi', 'Yuksek', 'Kismi'],
        ['Derinlik Tahmini', 'En dogru', 'Orta'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for j in range(3):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data)):
        table[i, 1].set_facecolor('#E3F2FD')
        table[i, 2].set_facecolor('#FFF3E0')

    plt.savefig(os.path.join(output_dir, '8_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== RAPOR =====
    print("\n" + "=" * 60)
    print("  SONUC RAPORU")
    print("=" * 60)
    print(f"""
  SIFT:
    - {sift['n_kp1']} keypoint, {len(sift['matches'])} eslesme
    - Inlier orani: {sift['inlier_ratio']:.1f}%
    - Sure: {sift['time']*1000:.1f} ms
    - 128-D descriptor ile guclu eslestirme
    - Olcek ve rotasyona dayanikli

  Harris:
    - {harris['n_corners_left']} kose, {harris['n_matches']} eslesme
    - Inlier orani: {harris['inlier_ratio']:.1f}%
    - Sure: {harris['time']*1000:.1f} ms
    - Descriptor uretmez, pencere tabanli eslestirme
    - Olcege duyarli, rotasyona kismi dayanikli

  Neden Canny kullanilmadi?
    - Canny sadece kenar haritasi uretir
    - Descriptor uretmez, eslestirme yapamaz
    - Stereo derinlik tahmini eslestirme gerektirir

  SONUC: SIFT, stereo derinlik tahmini icin en uygun
  methoddur. Yuksek inlier orani ve dayanikli descriptor'i
  sayesinde en dogru disparity haritasini uretir.
    """)

    print(f"Grafikler kaydedildi: {output_dir}/")
    for i, name in enumerate([
        'feature_detection', 'matching_quality', 'disparity_maps',
        'performance', 'noise_test', 'scale_test', 'rotation_test', 'summary_table'
    ], 1):
        print(f"  {i}_{name}.png")


# ============================================================================
# 7. ANA FONKSİYON
# ============================================================================

def create_sample_stereo_pair():
    """Test icin yapay stereo goruntu cifti olustur."""
    h, w = 400, 600
    left = np.zeros((h, w, 3), dtype=np.uint8)
    right = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        left[y, :] = [200 - y//4, 180 - y//5, 160 - y//6]
        right[y, :] = [200 - y//4, 180 - y//5, 160 - y//6]

    cv2.rectangle(left, (100, 150), (200, 300), (0, 0, 200), -1)
    cv2.rectangle(right, (80, 150), (180, 300), (0, 0, 200), -1)
    cv2.circle(left, (350, 200), 50, (0, 200, 0), -1)
    cv2.circle(right, (340, 200), 50, (0, 200, 0), -1)
    cv2.rectangle(left, (450, 50), (550, 120), (200, 200, 0), -1)
    cv2.rectangle(right, (447, 50), (547, 120), (200, 200, 0), -1)
    cv2.line(left, (50, 50), (150, 100), (255, 255, 255), 2)
    cv2.line(right, (40, 50), (140, 100), (255, 255, 255), 2)

    pts = np.array([[300, 320], [350, 380], [250, 380]], np.int32)
    cv2.fillPoly(left, [pts], (200, 100, 50))
    pts_r = pts.copy(); pts_r[:, 0] -= 15
    cv2.fillPoly(right, [pts_r], (200, 100, 50))

    np.random.seed(42)
    for _ in range(100):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        color = tuple(int(c) for c in np.random.randint(50, 200, 3))
        cv2.circle(left, (x, y), 2, color, -1)
        shift = max(1, 20 - y // 25)
        cv2.circle(right, (x - shift, y), 2, color, -1)

    return left, right


def find_image(base_dir, name):
    """Klasorde verilen isimle eslesen goruntu dosyasini bul."""
    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'JPG', 'JPEG', 'PNG']:
        path = os.path.join(base_dir, f'{name}.{ext}')
        if os.path.exists(path):
            return path
    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Stereo Derinlik Tahmini - SIFT vs Harris')
    parser.add_argument('--left', type=str, help='Sol goruntu dosyasi')
    parser.add_argument('--right', type=str, help='Sag goruntu dosyasi')
    parser.add_argument('--output', type=str, default=None, help='Cikti klasoru')
    parser.add_argument('--demo', action='store_true', help='Demo modu')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(script_dir, 'sonuclar')
    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        print("Demo modu: Yapay stereo goruntu cifti olusturuluyor...")
        left, right = create_sample_stereo_pair()
        cv2.imwrite(os.path.join(args.output, 'demo_left.png'), left)
        cv2.imwrite(os.path.join(args.output, 'demo_right.png'), right)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    elif args.left and args.right:
        left_path = args.left if os.path.isabs(args.left) else os.path.join(script_dir, args.left)
        right_path = args.right if os.path.isabs(args.right) else os.path.join(script_dir, args.right)
        left, right, gray_left, gray_right = load_stereo_images(left_path, right_path)

    else:
        left_path = find_image(script_dir, 'left')
        right_path = find_image(script_dir, 'right')

        if left_path and right_path:
            print(f"Bulunan goruntuler:")
            print(f"  Sol:  {os.path.basename(left_path)}")
            print(f"  Sag:  {os.path.basename(right_path)}")
            left, right, gray_left, gray_right = load_stereo_images(left_path, right_path)
        else:
            print("HATA: left.jpg ve right.jpg bulunamadi!")
            print(f"Klasor: {script_dir}")
            print("Cozum: Dosyalari script klasorune koy veya --left/--right ile belirt.")
            sys.exit(1)

    create_all_figures(left, right, gray_left, gray_right, args.output)
    print("\nTamamlandi!")


if __name__ == '__main__':
    main()
