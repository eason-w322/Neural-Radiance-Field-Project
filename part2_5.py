
import os, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt, imageio.v3 as iio
from tqdm import tqdm
from part2_1_rays import *
from part2_2_sampling import sample_along_rays
from part2_3_dataloader import RaysData
from part2_4_NeRF import NeRF


def volrend(sigmas, rgbs, step_size):
    alphas = 1 - torch.exp(-sigmas * step_size)             # [N, S, 1]

    # compute T using cumulative sum of log-transmittance
    eps = 1e-10
    trans = torch.cumsum(torch.log(torch.clamp(1 - alphas + eps, min=eps)), dim=1)
    T = torch.exp(torch.cat(
        [torch.zeros_like(trans[:, :1, :]), trans[:, :-1, :]], dim=1
    ))

    weights = T * alphas
    return torch.sum(weights * rgbs, dim=1)


def psnr(mse):
    return -10.0 * torch.log10(mse + 1e-8)


@torch.no_grad()
def render_full_image(model, K, c2w, H, W, n_samples, near, far, device):
    """Renders a full image by marching rays through every pixel."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32) + 0.5
    ray_o, ray_d = pixel_to_ray(K, c2w, uv)
    ray_o = np.broadcast_to(ray_o, ray_d.shape)

    pts_np, _ = sample_along_rays(
        ray_o, ray_d, n_samples=n_samples, near=near, far=far, perturb=False
    )
    pts = torch.tensor(pts_np, dtype=torch.float32, device=device)
    dirs = torch.tensor(ray_d, dtype=torch.float32, device=device)[:, None, :].expand_as(pts)

    rgb, sigma = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
    rgb = rgb.reshape(-1, n_samples, 3)
    sigma = sigma.reshape(-1, n_samples, 1)
    img = volrend(sigma, rgb, (far - near) / n_samples).reshape(H, W, 3).cpu().numpy()
    return np.clip(img, 0, 1)


def train_nerf(
    data_path="data/lego_200x200.npz",
    n_iters=1500,
    n_rays=10_000,
    n_samples=64,
    lr=5e-4,
    results_dir="results/nerf_training",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(results_dir, exist_ok=True)

    data = np.load(data_path)
    images_train = data["images_train"] / 255.0
    c2ws_train = data["c2ws_train"]
    images_val = data["images_val"] / 255.0
    c2ws_val = data["c2ws_val"]
    c2ws_test = data["c2ws_test"]
    focal = float(data["focal"])
    H, W = images_train.shape[1:3]
    K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=np.float32)

    dataset = RaysData(images_train, K, c2ws_train)

    model = NeRF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    near, far = 2.0, 6.0
    psnrs_train, psnrs_val, steps = [], [], []
    snapshots = {}

    print("\n🚀 Starting NeRF training...\n")

    # ---- Training Loop ----
    for it in tqdm(range(1, n_iters + 1)):
        model.train()
        rays_o, rays_d, gt_rgb = dataset.sample_rays(n_rays)
        rays_o = torch.tensor(rays_o, dtype=torch.float32, device=device)
        rays_d = F.normalize(torch.tensor(rays_d, dtype=torch.float32, device=device), dim=-1)
        gt_rgb = torch.tensor(gt_rgb, dtype=torch.float32, device=device)

        # Sample along rays
        pts_np, _ = sample_along_rays(
            rays_o.cpu().numpy(), rays_d.cpu().numpy(),
            n_samples=n_samples, near=near, far=far, perturb=True
        )
        pts = torch.tensor(pts_np, dtype=torch.float32, device=device)
        dirs = rays_d[:, None, :].expand_as(pts)

        # Forward pass
        rgb_pred, sigma_pred = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
        rgb_pred = rgb_pred.reshape(n_rays, n_samples, 3)
        sigma_pred = sigma_pred.reshape(n_rays, n_samples, 1)

        step_size = (far - near) / n_samples
        rendered = volrend(sigma_pred, rgb_pred, step_size)
        loss = criterion(rendered, gt_rgb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- Log Training PSNR ----
        if it % 10 == 0:
            psnr_val = psnr(loss).item()
            psnrs_train.append(psnr_val)
            steps.append(it)
            tqdm.write(f"[{it:04d}/{n_iters}] Loss={loss.item():.6f}, PSNR={psnr_val:.2f}dB")

        # ---- Validation Evaluation ----
        if it % 100 == 0:
            model.eval()
            val_scores = []
            with torch.no_grad():
                for i in range(min(6, len(images_val))):
                    img_pred = render_full_image(model, K, c2ws_val[i], H, W, n_samples, near, far, device)
                    img_gt = images_val[i]
                    mse_val = np.mean((img_pred - img_gt) ** 2)
                    val_scores.append(-10 * np.log10(mse_val + 1e-8))
            mean_val_psnr = np.mean(val_scores)
            psnrs_val.append(mean_val_psnr)
            tqdm.write(f"Validation PSNR: {mean_val_psnr:.2f} dB")

        # ---- Save Snapshots ----
        if it in [50, 100, 250, 500, 1000, 1500]:
            img = render_full_image(model, K, c2ws_val[0], H, W, n_samples, near, far, device)
            plt.imsave(f"{results_dir}/iter_{it:04d}.png", img)
            snapshots[it] = img


    plt.figure(figsize=(7, 5))
    plt.plot(steps, psnrs_train, label="Training PSNR", marker="o")
    plt.plot(np.arange(100, n_iters + 1, 100), psnrs_val, label="Validation PSNR", marker="s", color="orange")
    plt.title("Training & Validation PSNR Curves")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/psnr_curves.png", dpi=200)
    plt.close()


    if snapshots:
        ncols = len(snapshots)
        plt.figure(figsize=(3 * ncols, 3))
        for i, (k, img) in enumerate(snapshots.items()):
            plt.subplot(1, ncols, i + 1)
            plt.imshow(img)
            plt.title(f"Iter {k}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/training_progress.png", dpi=200)
        plt.close()


    print("🎥 Rendering novel views for spherical orbit...")
    frames = []
    with torch.no_grad():
        for idx, c2w in enumerate(tqdm(c2ws_test[::2])):  # ~50 frames
            img = render_full_image(model, K, c2w, H, W, n_samples, near, far, device)
            frames.append((img * 255).astype(np.uint8))

    gif_path = f"{results_dir}/lego_rotation.gif"
    iio.imwrite(gif_path, frames, format="GIF", fps=10, loop = 0)
    print(f"✅ Saved novel-view rotation GIF → {gif_path}")
    print(f"Final Training PSNR: {psnrs_train[-1]:.2f} dB")
    print(f"Final Validation PSNR: {psnrs_val[-1]:.2f} dB")

    return model, psnrs_train, psnrs_val


if __name__ == "__main__":
    torch.manual_seed(42)
    sigmas = torch.rand((10, 64, 1))
    rgbs = torch.rand((10, 64, 3))
    step_size = (6.0 - 2.0) / 64
    rendered_colors = volrend(sigmas, rgbs, step_size)
    correct = torch.tensor([
        [0.5006, 0.3728, 0.4728],
        [0.4322, 0.3559, 0.4134],
        [0.4027, 0.4394, 0.4610],
        [0.4514, 0.3829, 0.4196],
        [0.4002, 0.4599, 0.4103],
        [0.4471, 0.4044, 0.4069],
        [0.4285, 0.4072, 0.3777],
        [0.4152, 0.4190, 0.4361],
        [0.4051, 0.3651, 0.3969],
        [0.3253, 0.3587, 0.4215]
    ])
    assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)
    print("✅ volrend() passed reference test!\n")

    # Start training
    train_nerf()