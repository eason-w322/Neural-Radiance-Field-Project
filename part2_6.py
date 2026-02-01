import os, math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from tqdm import tqdm
import cv2

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🔥 Using Apple MPS accelerator")
else:
    device = torch.device("cpu")
    print("⚠️ MPS unavailable — using CPU (slower).")

# ============================================================
#                    PIXEL → RAY  (FAST VERSION)
# ============================================================
def transform_points(c2w: np.ndarray, x_c: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return (x_c @ R.T) + t


def pixel_to_camera(K: np.ndarray, uv: np.ndarray, s: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx * s
    y = (uv[:, 1] - cy) / fy * s
    return np.stack([x, y, s], axis=-1)


def pixel_to_ray(K: np.ndarray, c2w: np.ndarray, uv: np.ndarray):
    o = c2w[:3, 3].reshape(1, 3)
    xc = pixel_to_camera(K, uv, s=np.ones((uv.shape[0],), dtype=np.float32))
    xw = transform_points(c2w, xc)
    d = xw - o
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    o = np.repeat(o, repeats=uv.shape[0], axis=0)
    return o.astype(np.float32), d.astype(np.float32)


# ============================================================
#                    TORCH SAMPLING (FAST)
# ============================================================
def sample_along_rays(rays_o, rays_d,
                      n_samples, near, far,
                      perturb: bool,
                      training: bool,
                      device,
                      jitter_scale: float = 1.0):

    B = rays_o.shape[0]
    t_vals = torch.linspace(near, far, steps=n_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(B, -1)

    if perturb and training:
        mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        lower = torch.cat([t_vals[:, :1], mids], dim=1)
        upper = torch.cat([mids, t_vals[:, -1:]], dim=1)
        t_rand = torch.rand_like(t_vals) * jitter_scale
        t_vals = lower + (upper - lower) * t_rand

    pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[..., None]
    dirs = rays_d[:, None, :].expand_as(pts)
    return t_vals, pts, dirs


# ============================================================
#                  NeRF MODEL (FAST VERSION)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, in_dim: int, L: int):
        super().__init__()
        freqs = (2.0 ** torch.arange(L, dtype=torch.float32)) * (2.0 * math.pi)
        self.register_buffer("freqs", freqs, persistent=False)
        self.in_dim = in_dim
        self.L = L
        self.out_dim = in_dim + 2 * in_dim * L

    def forward(self, x):
        outs = [x]
        for k in range(self.L):
            w = self.freqs[k]
            outs.append(torch.sin(x * w))
            outs.append(torch.cos(x * w))
        return torch.cat(outs, dim=-1)


class NeRF(nn.Module):
    def __init__(self, L_xyz=10, L_dir=4, width=256, depth=8, skip_at=4):
        super().__init__()
        self.pe_xyz = PositionalEncoding(3, L_xyz)
        self.pe_dir = PositionalEncoding(3, L_dir)

        in_xyz = self.pe_xyz.out_dim
        in_dir = self.pe_dir.out_dim
        self.skip_at = skip_at
        self.depth = depth

        layers1 = []
        last_dim = in_xyz
        for _ in range(skip_at):
            layers1.append(nn.Linear(last_dim, width))
            layers1.append(nn.ReLU(inplace=True))
            last_dim = width
        self.block1 = nn.Sequential(*layers1)

        layers2 = []
        last_dim = width + in_xyz
        for _ in range(depth - skip_at):
            layers2.append(nn.Linear(last_dim, width))
            layers2.append(nn.ReLU(inplace=True))
            last_dim = width
        self.block2 = nn.Sequential(*layers2)

        self.fc_sigma = nn.Linear(width, 1)
        self.fc_feat = nn.Linear(width, width)

        self.fc_color = nn.Sequential(
            nn.Linear(width + in_dir, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, x, d):
        x_enc = self.pe_xyz(x)
        d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-9)
        d_enc = self.pe_dir(d)

        h = self.block1(x_enc)
        h = torch.cat([h, x_enc], dim=-1)
        h = self.block2(h)

        sigma = F.relu(self.fc_sigma(h))
        feat = self.fc_feat(h)
        rgb = self.fc_color(torch.cat([feat, d_enc], dim=-1))
        return rgb, sigma


# ============================================================
#                 VOLUME RENDERING
# ============================================================
def volume_render(sigmas, rgbs, t_vals):
    B, S = t_vals.shape
    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    deltas = torch.cat([deltas, deltas[:, -1:]], dim=1)
    sigma = sigmas.squeeze(-1)
    alpha = 1.0 - torch.exp(-sigma * deltas)

    T = torch.cumprod(
        torch.cat([torch.ones((B, 1), device=sigmas.device),
                   1.0 - alpha + 1e-10], dim=1),
        dim=1
    )[:, :-1]

    weights = alpha * T
    comp_rgb = torch.sum(weights[..., None] * rgbs, dim=1)
    return comp_rgb


# ============================================================
#               FULL IMAGE RENDERING
# ============================================================
@torch.no_grad()
def render_full_image(model, K, c2w, H, W,
                      n_samples, near, far,
                      device, chunk=8192):

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32) + 0.5

    ro_np, rd_np = pixel_to_ray(K, c2w, uv)
    ro = torch.as_tensor(ro_np, dtype=torch.float32, device=device)
    rd = torch.as_tensor(rd_np, dtype=torch.float32, device=device)

    out_rgb = []

    for i in range(0, ro.shape[0], chunk):
        j = min(i + chunk, ro.shape[0])
        t_vals, pts, dirs = sample_along_rays(
            ro[i:j], rd[i:j], n_samples,
            near, far,
            perturb=False, training=False,
            device=device
        )

        B = pts.shape[0]
        x = pts.reshape(B * n_samples, 3)
        d = dirs.reshape(B * n_samples, 3)

        rgb, sigma = model(x, d)
        rgb = rgb.reshape(B, n_samples, 3)
        sigma = sigma.reshape(B, n_samples, 1)

        comp_rgb = volume_render(sigma, rgb, t_vals)
        out_rgb.append(comp_rgb)

    rgb_full = torch.cat(out_rgb, dim=0).reshape(H, W, 3)
    return rgb_full.clamp(0, 1).cpu().numpy()


# ============================================================
#              SIMPLE PSNR FUNCTION
# ============================================================
def psnr_from_mse(mse):
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


# ============================================================
#                   DATASET SAMPLER
# ============================================================
class RaysData:
    def __init__(self, images, K, c2ws):
        self.images = images.astype(np.float32)
        self.K = K
        self.c2ws = c2ws
        self.H, self.W = images.shape[1:3]
        self.N = images.shape[0]

    def sample_rays(self, n_rays):
        img_inds = np.random.randint(0, self.N, size=n_rays)
        u = np.random.randint(0, self.W, size=n_rays)
        v = np.random.randint(0, self.H, size=n_rays)

        uv = np.stack([u + 0.5, v + 0.5], axis=-1).astype(np.float32)

        rays_o = np.zeros((n_rays, 3), dtype=np.float32)
        rays_d = np.zeros((n_rays, 3), dtype=np.float32)
        rgbs = np.zeros((n_rays, 3), dtype=np.float32)

        for i in range(n_rays):
            idx = img_inds[i]
            c2w = self.c2ws[idx]
            ro, rd = pixel_to_ray(self.K, c2w, uv[i:i+1])
            rays_o[i] = ro
            rays_d[i] = rd
            rgbs[i] = self.images[idx, int(v[i]), int(u[i])]

        rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)
        return rays_o, rays_d, rgbs


# ============================================================
#                   TRAINING LOOP
# ============================================================
def train_nerf(
    data_path="data/my_data.npz",
    n_iters=10000,
    n_rays=4096,
    n_samples=64,
    lr=3e-4,
    near=0.1,
    far=1.5,
    device = "mps" if torch.backends.mps.is_available() else "cpu",
    results_dir="results/my_nerf_training"
):
    os.makedirs(results_dir, exist_ok=True)

    data = np.load(data_path)
    images_train = data["images_train"] / 255.0
    images_val = data["images_val"] / 255.0
    c2ws_train = data["c2ws_train"]
    c2ws_val = data["c2ws_val"]

    # Use resized K from dataset if saved
    if "K" in data:
        K = data["K"].astype(np.float32)
        focal = float(K[0,0])
    else:
        focal = float(data["focal"])
        H, W = images_train.shape[1:3]
        K = np.array([[focal, 0, W/2],
                      [0, focal, H/2],
                      [0, 0, 1]], dtype=np.float32)

    H, W = images_train.shape[1:3]
    print(f"Training images: {H}×{W}, K loaded.")
    print("Device:", device)

    dataset = RaysData(images_train, K, c2ws_train)
    model = NeRF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    psnrs, losses, steps = [], [], []

    for it in tqdm(range(1, n_iters + 1)):
        model.train()

        ro_np, rd_np, gt_np = dataset.sample_rays(n_rays)
        ro = torch.tensor(ro_np, dtype=torch.float32, device=device)
        rd = torch.tensor(rd_np, dtype=torch.float32, device=device)
        gt = torch.tensor(gt_np, dtype=torch.float32, device=device)

        t_vals, pts, dirs = sample_along_rays(
            ro, rd, n_samples,
            near, far,
            perturb=True, training=True,
            device=device
        )

        B = n_rays
        x = pts.reshape(B * n_samples, 3)
        d = dirs.reshape(B * n_samples, 3)

        rgb_pred, sigma_pred = model(x, d)
        rgb_pred = rgb_pred.reshape(B, n_samples, 3)
        sigma_pred = sigma_pred.reshape(B, n_samples, 1)

        rgb_final = volume_render(sigma_pred, rgb_pred, t_vals)
        loss = criterion(rgb_final, gt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 10 == 0:
            mse = loss.item()
            psnr = psnr_from_mse(mse)
            psnrs.append(psnr)
            losses.append(mse)
            steps.append(it)
            tqdm.write(f"[{it}] loss={mse:.6f}, PSNR={psnr:.2f} dB")

        if it in [200, 500, 1000, 1500, 2000, 2500]:
            with torch.no_grad():
                img = render_full_image(model, K, c2ws_val[1],
                                        H, W,
                                        n_samples, near, far,
                                        device)
                plt.imsave(f"{results_dir}/val_{it:04d}.png", img)

    # CURVES
    plt.figure(figsize=(6,4))
    plt.plot(steps, psnrs); plt.xlabel("Iter"); plt.ylabel("PSNR")
    plt.grid(); plt.tight_layout()
    plt.savefig(f"{results_dir}/psnr_curve.png"); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(steps, losses); plt.xlabel("Iter"); plt.ylabel("Loss")
    plt.grid(); plt.tight_layout()
    plt.savefig(f"{results_dir}/loss_curve.png"); plt.close()

    print("\n==================== GIF GENERATION ====================\n")

    # ------------------------------------------------------------
    # Shared: Look-at function
    # ------------------------------------------------------------
    def look_at(eye, target, up=np.array([0, 1, 0], dtype=np.float32)):
        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        right = np.cross(up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)

        up2 = np.cross(forward, right)
        up2 = up2 / (np.linalg.norm(up2) + 1e-8)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up2
        c2w[:3, 2] = forward
        c2w[:3, 3] = eye
        return c2w

    # ------------------------------------------------------------
    # Compute unified scene center + radius vector
    # ------------------------------------------------------------
    scene_center = c2ws_train[:, :3, 3].mean(axis=0)
    base_eye = c2ws_val[0][:3, 3]           # first validation camera
    radius_vec = base_eye - scene_center    # vector from center to camera
    shrink = 0.2                           # shrink radius for cleaner GIF

    # ------------------------------------------------------------
    # Rotation matrices for each axis
    # ------------------------------------------------------------

    def rot_z(theta):
        return np.array([
            [ np.cos(theta), -np.sin(theta), 0 ],
            [ np.sin(theta),  np.cos(theta), 0 ],
            [ 0,              0,             1 ],
        ], dtype=np.float32)

    # ============================================================
    # 1) BEST WORKING ORBIT (VAL CAMERA–BASED ORBIT)
    # ============================================================
    print("🎥 Generating orbit.gif (VAL based) ...")

    base_c2w = c2ws_val[1]
    val_center = base_c2w[:3, 3] + base_c2w[:3, 2] * 0.25  # original target

    frames_orbit = []
    for i in tqdm(range(20), desc="VAL Orbit"):
        theta = 2 * np.pi * i / 20

        v = base_eye - val_center
        v_rot = rot_z(theta) @ v
        new_eye = val_center + shrink * v_rot

        c2w_new = look_at(new_eye, val_center)
        img = render_full_image(model, K, c2w_new, H, W, n_samples, near, far, device)
        frames_orbit.append((img * 255).astype(np.uint8))

    iio.imwrite(f"{results_dir}/orbit.gif", frames_orbit, format="GIF", loop=0, fps=8)
    print("✔ Saved → orbit.gif\n")


    print("🎥 Generating true_object_orbit.gif ...")

    scene_center = np.array([0.0448737, -0.12209456, 0.28780879], dtype=np.float32)

    def normalize(v):
        return v / (np.linalg.norm(v) + 1e-8)

    # use any training view to determine a stable world up-direction
    up = normalize(c2ws_train[0][:3, 1])

    # orbit radius
    radius = 0.25
    height = 0.10      # small elevation to avoid tilting
    n_frames = 20

    frames = []
    for i in tqdm(range(n_frames), desc="Object Orbit"):

        theta = 2 * np.pi * i / n_frames

        # Camera position moving around *object center*
        eye = scene_center + np.array([
            radius * math.cos(theta),
            height,
            radius * math.sin(theta)
        ], dtype=np.float32)

        # --- build a proper look-at matrix ---
        forward = normalize(scene_center - eye)
        right = normalize(np.cross(up, forward))
        up_new = normalize(np.cross(forward, right))

        c2w_new = np.eye(4, dtype=np.float32)
        c2w_new[:3, 0] = right
        c2w_new[:3, 1] = up_new
        c2w_new[:3, 2] = forward
        c2w_new[:3, 3] = eye

        # render
        img = render_full_image(model, K, c2w_new, H, W,
                                n_samples, near, far, device)
        frames.append((img * 255).astype(np.uint8))

    iio.imwrite(f"{results_dir}/true_object_orbit.gif",
                frames, format="GIF", fps=8, loop=0)

    print("✔ Saved → true_object_orbit.gif")
    return model, K, H, W, c2ws_val




if __name__ == "__main__":
    torch.manual_seed(42)
    model, K, H, W, c2ws_val = train_nerf()