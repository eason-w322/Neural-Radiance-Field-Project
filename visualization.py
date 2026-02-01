import cv2
import numpy as np
import glob
import os
import viser
import time
from PIL import Image, ImageOps

from part2_1_rays import pixel_to_ray
from part2_2_sampling import sample_along_rays


# ============================================================
# Step 1 — Inspect image resolutions
# ============================================================
def check_image_shapes(folder):
    image_list = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    shapes = {}

    for p in image_list:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img)
        arr = np.array(img)

        shapes.setdefault(arr.shape, []).append(os.path.basename(p))

    print("\n=== Image Shape Report ===")
    for shape, files in shapes.items():
        print(f"{shape}: {len(files)} images")
        for f in files:
            print("   ", f)

    if len(shapes) == 1:
        print("All images have consistent shape ✓")
    else:
        print("❌ Inconsistent shapes detected")



# ============================================================
# Step 2 — Load + filter Labubu images
# ============================================================
image_dir = "data/labubu2_images"
check_image_shapes(image_dir)

raw_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
filtered_paths = []

for path in raw_paths:
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)
    arr = np.array(pil_img)

    # Remove wrong orientation (4032×3024)
    if arr.shape[:2] == (4032, 3024):
        print(f"[Drop] {os.path.basename(path)} shape={arr.shape}")
        continue

    filtered_paths.append(path)

print(f"\nUsing {len(filtered_paths)} images after filtering.\n")



# ============================================================
# Step 3 — Load intrinsics
# ============================================================
calib_file = "data/camera_calib.npz"

with np.load(calib_file) as X:
    K = X["K"]
    dist = X["dist"]

print("Loaded calibration:")
print("K =\n", K)
print("dist =", dist.ravel())



# ============================================================
# Step 4 — ArUco setup
# ============================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

tag_size = 0.057  # meter

object_points = np.array([
    [0, 0, 0],
    [tag_size, 0, 0],
    [tag_size, tag_size, 0],
    [0, tag_size, 0]
], dtype=np.float32)



# ============================================================
# Step 5 — Solve camera poses
# ============================================================
c2ws = []
valid_imgs = []

for path in filtered_paths:
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)
    img = np.array(pil_img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is None or len(corners) == 0:
        print(f"[Skip] No ArUco tag in {os.path.basename(path)}")
        continue

    image_points = corners[0].reshape(-1, 2)

    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist
    )
    if not success:
        print(f"[Skip] solvePnP failed for {os.path.basename(path)}")
        continue

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)

    # World-to-camera
    w2c_R = R
    w2c_t = t

    # Camera-to-world
    c2w = np.eye(4)
    c2w[:3, :3] = w2c_R.T
    c2w[:3, 3] = (-w2c_R.T @ w2c_t).flatten()

    # NeRF coordinate fix
    D = np.diag([1, -1, -1])
    c2w[:3, :3] = D @ c2w[:3, :3]
    c2w[:3, 3] = D @ c2w[:3, 3]

    c2ws.append(c2w)
    valid_imgs.append(img)

print(f"\nSuccessfully estimated {len(c2ws)} camera poses.\n")



# ============================================================
# Step 6 — Ray sampling (GOOD VERSION — from multiple poses)
# ============================================================
def sample_rays_from_multiple_cameras(c2ws, images, K, N_RAYS=200):
    rays_o_list = []
    rays_d_list = []
    pts_list = []

    n_images = len(c2ws)

    for _ in range(N_RAYS):
        # pick a random camera
        img_idx = np.random.randint(0, n_images)
        img = images[img_idx]
        c2w = c2ws[img_idx]

        H, W = img.shape[:2]

        # random pixel
        u = np.random.randint(0, W)
        v = np.random.randint(0, H)
        uv = np.array([[u + 0.5, v + 0.5]], dtype=np.float32)

        # shoot ray
        ray_o, ray_d = pixel_to_ray(K, c2w, uv)
        ray_o = np.broadcast_to(ray_o, ray_d.shape)

        rays_o_list.append(ray_o[0])
        rays_d_list.append(ray_d[0])

        # sample points
        pts, _ = sample_along_rays(ray_o, ray_d, n_samples=32, near=2, far=5, perturb=True)
        pts_list.append(pts[0])

    return (
        np.vstack(rays_o_list),
        np.vstack(rays_d_list),
        np.stack(pts_list, axis=0)
    )



# ============================================================
# Step 7 — Visualize using VISER
# ============================================================
if len(c2ws):
    H, W = valid_imgs[0].shape[:2]

    rays_o, rays_d, pts = sample_rays_from_multiple_cameras(c2ws, valid_imgs, K)

    server = viser.ViserServer(share=True)
    print("Launching VISER viewer...\n")

    # camera frustums
    for i, (img, c2w) in enumerate(zip(valid_imgs, c2ws)):
        server.scene.add_camera_frustum(
            f"/cams/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.01,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=img
        )

    # rays
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=np.stack((o, o + d * 0.3)),
        )

    # sample points
    server.scene.add_point_cloud(
        "/samples",
        points=pts.reshape(-1, 3),
        colors=np.zeros_like(pts).reshape(-1, 3),
        point_size=0.01,
    )

    print("VISER is ready! (waiting loop)")
    while True:
        time.sleep(0.1)

else:
    print("No camera poses to visualize.")