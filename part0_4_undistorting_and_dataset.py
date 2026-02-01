import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

calib_file = "data/camera_calib.npz"
pose_image_file = "data/poses_and_images.npz"
output_dataset_file = "data/my_data.npz"
undistorted_dir = "results/undistorted_preview"


# ================================================================
# Step 1 — Load camera calibration (full-res)
# ================================================================
with np.load(calib_file) as X:
    K_full = X["K"].astype(np.float64)
    dist   = X["dist"].astype(np.float64)

print("Loaded full-resolution calibration:")
print("K_full =\n", K_full)
print("dist   =", dist.ravel())


# ================================================================
# Step 2 — Load resized images + resized K from Part 0.3
# ================================================================
if not os.path.exists(pose_image_file):
    raise FileNotFoundError(
        "poses_and_images.npz not found — run Part 0.3 first."
    )

with np.load(pose_image_file) as data:
    c2ws   = data["c2ws"].astype(np.float64)     # (N,4,4)
    images = data["images"].astype(np.uint8)     # (N,H,W,3)
    K      = data["K"].astype(np.float64)        # <-- already scaled!

print(f"\nLoaded {len(images)} resized images and poses.")
print("Resized K =\n", K)


# ================================================================
# Step 3 — Undistort resized images
# ================================================================
os.makedirs(undistorted_dir, exist_ok=True)
undistorted_images = []

print("\nUndistorting resized images...")
for i, img in enumerate(images):

    # OpenCV expects RGB->BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    undistorted_bgr = cv2.undistort(img_bgr, K, dist)
    undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)

    undistorted_images.append(undistorted_rgb)

    cv2.imwrite(
        os.path.join(undistorted_dir, f"undistorted_{i:03d}.jpg"),
        undistorted_bgr,
    )

undistorted_images = np.array(undistorted_images, dtype=np.uint8)
print(f"Saved {len(undistorted_images)} undistorted images → {undistorted_dir}")


# ================================================================
# Step 4 — Train/Val/Test split
# ================================================================
n = len(undistorted_images)
idx = np.arange(n)

# First split: 98% train, 2% temp
train_idx, val_idx = train_test_split(
    idx,
    test_size=0.1,       # 10% validation
    shuffle=True,
    random_state=42
)

def subset(arr, inds):
    return arr[inds]

images_train = subset(undistorted_images, train_idx)
c2ws_train   = subset(c2ws, train_idx)

images_val = subset(undistorted_images, val_idx)
c2ws_val   = subset(c2ws, val_idx)

# No test set (NeRF doesn't really need it)
images_test = np.empty((0,), dtype=object)
c2ws_test   = np.empty((0,), dtype=object)

print("\nDataset split (95/5):")
print(f"  Train: {len(train_idx)}")
print(f"  Val:   {len(val_idx)}")
print(f"  Test:  {len(images_test)}  (disabled)")


# ================================================================
# Step 5 — Use focal length from scaled K
# ================================================================
focal = float((K[0, 0] + K[1, 1]) / 2)
print(f"\nFocal length (resized): {focal:.3f}")


# ================================================================
# Step 6 — Save dataset for NeRF training
# ================================================================
np.savez(
    output_dataset_file,
    images_train=images_train,
    c2ws_train=c2ws_train,
    images_val=images_val,
    c2ws_val=c2ws_val,
    focal=focal,
    K=K,
)

print(f"\nDataset saved → {output_dataset_file}")
print("Your dataset is ready for NeRF training 🎉")