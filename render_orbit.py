import numpy as np
import torch, imageio.v3 as iio
from tqdm import tqdm
import cv2

from part2_1_rays import pixel_to_ray
from part2_2_sampling import sample_along_rays
from part2_4_NeRF import NeRFNetwork
from part2_6 import render_full_image   # or paste render_full_image here


# --------------------------------------------------------------
# Load trained model
# --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

results_dir = "results/my_nerf_training"
model_path = f"{results_dir}/nerf_model.pth"
data_path = "data/my_data.npz"

print("📂 Loading data...")
data = np.load(data_path)

images_train = data["images_train"] / 255.0
c2ws_train = data["c2ws_train"]
focal = float(data["focal"])

H0, W0 = images_train.shape[1:3]
H, W = H0, W0  # whatever resolution your training used

K = np.array([
    [focal, 0, W / 2],
    [0, focal, H / 2],
    [0, 0, 1]
], dtype=np.float32)

print("📐 Intrinsic matrix:\n", K)


# --------------------------------------------------------------
# Rebuild the NeRF model and load weights
# --------------------------------------------------------------
print("🔧 Initializing NeRF model...")
model = NeRFNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Loaded trained model weights.")


# --------------------------------------------------------------
# Generate LOOK-AT orbit poses
# --------------------------------------------------------------
def normalize(v):
    return v / np.linalg.norm(v)

def generate_lookat_orbit(c2w_ref, n_frames=40, radius=0.12):
    poses = []

    cam_pos = c2w_ref[:3, 3]
    forward = c2w_ref[:3, 2]
    obj_center = cam_pos + forward * 0.3     # estimate focus point

    up = np.array([0, 1, 0], dtype=np.float32)
    forward_dir = normalize(obj_center - cam_pos)
    right = normalize(np.cross(forward_dir, up))
    up = normalize(np.cross(right, forward_dir))

    for i in range(n_frames):
        theta = 2 * np.pi * (i / n_frames)
        offset = np.cos(theta) * radius * right + np.sin(theta) * radius * up
        new_pos = obj_center + offset

        new_forward = normalize(obj_center - new_pos)
        new_right = normalize(np.cross(up, new_forward))
        new_up = normalize(np.cross(new_forward, new_right))

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = new_right
        pose[:3, 1] = new_up
        pose[:3, 2] = new_forward
        pose[:3, 3] = new_pos
        poses.append(pose)

    return poses


ref_pose = c2ws_train[len(c2ws_train)//2]
orbit_poses = generate_lookat_orbit(ref_pose, n_frames=40)


# --------------------------------------------------------------
# Render GIF using the loaded model
# --------------------------------------------------------------
frames = []
print("🎥 Rendering orbit frames...")

for c2w in tqdm(orbit_poses):
    img = render_full_image(
        model, K, c2w, H, W,
        n_samples=64, near=0.02, far=0.5,
        device=device
    )
    frames.append((img * 255).astype(np.uint8))

gif_path = f"{results_dir}/orbit_lookat.gif"
iio.imwrite(gif_path, frames, format="GIF", fps=8, loop=0)

print("✅ Saved:", gif_path)