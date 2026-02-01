import cv2
import numpy as np
import glob
import os
import viser
import time
from PIL import Image, ImageOps   # <-- EXIF transpose ON

calib_file = "data/camera_calib.npz"
image_dir  = "data/labubu2_images"

# ================================
# Target resized resolution
# Original: H0 = 3024, W0 = 4032
# ================================
TARGET_H = 600
TARGET_W = 800

# ================================================================
# Step 1 — Check original image shapes (EXIF-corrected)
# ================================================================
def check_image_shapes(folder):
    image_list = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    shapes = {}
    for p in image_list:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img)  # IMPORTANT
        arr = np.array(img)
        shapes.setdefault(arr.shape, []).append(os.path.basename(p))

    print("=== Image Shape Report ===")
    for shape, files in shapes.items():
        print(f"Shape {shape}: {len(files)} images")

    if len(shapes) == 1:
        print("All images have consistent shape ✓\n")
    else:
        print("❌ WARNING: Inconsistent shapes detected!\n")

check_image_shapes(image_dir)


# ================================================================
# Step 2 — Load ALL images with EXIF transpose + Resize FIRST
# ================================================================
raw_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
images_resized = []

for path in raw_paths:
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)  # correct orientation
    img = np.array(pil_img)

    H0, W0 = img.shape[:2]

    scale_h = TARGET_H / H0
    scale_w = TARGET_W / W0
    scale = min(scale_h, scale_w)   # preserve AR

    new_H = int(H0 * scale)
    new_W = int(W0 * scale)

    img_resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
    images_resized.append(img_resized)

images_resized = np.array(images_resized, dtype=np.uint8)
print(f"Resized all images → {new_H}×{new_W}")

# ================================================================
# Step 3 — Load intrinsics and scale them
# ================================================================
with np.load(calib_file) as X:
    K_full = X["K"].astype(np.float64)
    dist   = X["dist"].astype(np.float64)

print("Loaded full-res K:\n", K_full)

# scale factors from original → resized
scale_factor = new_W / W0   # same as new_H / H0

fx_new = K_full[0,0] * scale_factor
fy_new = K_full[1,1] * scale_factor
cx_new = K_full[0,2] * scale_factor
cy_new = K_full[1,2] * scale_factor

K = np.array([
    [fx_new, 0,     cx_new],
    [0,     fy_new, cy_new],
    [0,     0,      1]
], dtype=np.float64)

print("Scaled K:\n", K)

# ================================================================
# Step 4 — ArUco config
# ================================================================
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

tag_size = 0.057  # meters
object_points = np.array([
    [0, 0, 0],
    [tag_size, 0, 0],
    [tag_size, tag_size, 0],
    [0, tag_size, 0]
], dtype=np.float64)


# ================================================================
# Step 5 — Pose estimation on RESIZED images
# ================================================================
c2ws = []
valid_imgs = []

for img in images_resized:

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is None or len(corners) == 0:
        print("[Skip] No ArUco tag")
        continue

    image_points = corners[0].reshape(-1, 2).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist
    )
    if not success:
        print("[Skip] solvePnP failed")
        continue

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    # world → camera
    w2c = np.eye(4)
    w2c[:3,:3] = R
    w2c[:3, 3] = t

    # camera → world
    c2w = np.eye(4)
    c2w[:3,:3] = R.T
    c2w[:3, 3] = -R.T @ t

    # NeRF coordinate conversion (OpenCV → NeRF)
    D = np.diag([1, -1, -1])
    c2w[:3,:3] = D @ c2w[:3,:3]
    c2w[:3, 3] = D @ c2w[:3, 3]

    c2ws.append(c2w)
    valid_imgs.append(img)

print(f"\nSuccessfully estimated poses for {len(c2ws)} images.")


# ================================================================
# Step 6 — Save dataset
# ================================================================
save_path = "data/poses_and_images.npz"
np.savez(
    save_path,
    c2ws=np.array(c2ws, dtype=np.float64),
    images=np.array(valid_imgs, dtype=np.uint8),
    K=K.astype(np.float64)
)
print(f"\nSaved poses + resized images → {save_path}")


# ================================================================
# Step 7 — Viser Visualization
# ================================================================
if len(c2ws):
    H, W = valid_imgs[0].shape[:2]

    server = viser.ViserServer(share=True)
    print("Launching Viser viewer...")

    for i, (img, c2w) in enumerate(zip(valid_imgs, c2ws)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0,0]),
            aspect=W / H,
            scale=0.01,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3,:3]).wxyz,
            position=c2w[:3, 3],
            image=img
        )

    while True:
        time.sleep(0.1)

else:
    print("No valid poses to visualize.")