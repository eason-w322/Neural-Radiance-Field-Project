import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs) * math.pi

    def forward(self, x):
        xb = x[..., None] * self.freq_bands
        sin, cos = torch.sin(xb), torch.cos(xb)
        return torch.cat([x, sin.reshape(x.shape[0], -1), cos.reshape(x.shape[0], -1)], dim=-1)


class NeuralField2D(nn.Module):
    def __init__(self, num_freqs=10, hidden_dim=256):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs)
        in_dim = 2 + 4 * num_freqs  # (x,y) + sin/cos pairs

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()  # RGB ∈ [0,1]
        )

    def forward(self, x):
        x_encoded = self.pe(x)
        return self.layers(x_encoded)


def psnr(mse):
    return -10.0 * torch.log10(mse)

def load_image(path):
    img = imageio.v3.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img

def prepare_data(img):
    H, W, _ = img.shape
    y, x = torch.meshgrid(
        torch.linspace(0, 1, H),
        torch.linspace(0, 1, W),
        indexing="ij"
    )
    coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
    rgbs = torch.tensor(img.reshape(-1, 3), dtype=torch.float32)
    return coords, rgbs, H, W

def train_neural_field(
    image_path,
    num_freqs=10,
    hidden_dim=256,
    batch_size=10000,
    lr=1e-2,
    iters=3000,
    results_dir="results/Neural Field to a 2D Image",
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_progress=False,
):
    os.makedirs(results_dir, exist_ok=True)
    model_name = f"freq{num_freqs}_width{hidden_dim}"

    img = load_image(image_path)
    coords, rgbs, H, W = prepare_data(img)
    coords, rgbs = coords.to(device), rgbs.to(device)

    model = NeuralField2D(num_freqs=num_freqs, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    snapshots = {}
    psnr_values = []
    psnr_steps = []

    save_iters = [50, 100, 250, 500, 1000, 2000]

    for it in tqdm(range(iters + 1)):
        inds = torch.randint(0, coords.shape[0], (batch_size,))
        batch_x, batch_y = coords[inds], rgbs[inds]

        pred_rgb = model(batch_x)
        loss = criterion(pred_rgb, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if save_progress and it % 2 == 0:
            with torch.no_grad():
                psnr_val = psnr(loss).item()
                psnr_values.append(psnr_val)
                psnr_steps.append(it)


        if save_progress and it in save_iters:
            with torch.no_grad():
                pred_full = model(coords).reshape(H, W, 3).cpu().numpy()
            snapshots[it] = pred_full


    with torch.no_grad():
        pred_full = model(coords).reshape(H, W, 3).cpu().numpy()


    if save_progress and len(snapshots) > 0:
        steps = save_iters
        ncols = len(steps) + 1  # include original image
        plt.figure(figsize=(3 * ncols, 3))
        plt.suptitle(f"Training Progression (num_freqs={num_freqs}, width={hidden_dim})")

        # Plot original
        plt.subplot(1, ncols, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        # Reconstructions
        for i, step in enumerate(steps):
            plt.subplot(1, ncols, i + 2)
            plt.imshow(snapshots[step])
            plt.title(f"Iter {step}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{results_dir}/progression_freq{num_freqs}_width{hidden_dim}.png", dpi=200)
        plt.close()

    # Save PSNR curve
    if save_progress and len(psnr_values) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(psnr_steps, psnr_values, marker='o')
        plt.title(f"PSNR Curve (num_freqs={num_freqs}, width={hidden_dim})")
        plt.xlabel("Iteration")
        plt.ylabel("PSNR (dB)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/psnr_curve_freq{num_freqs}_width{hidden_dim}.png", dpi=200)
        plt.close()

    return pred_full


if __name__ == "__main__":
    image_path = "data/flowers.jpg"
    freqs_list = [2, 10]
    widths_list = [64, 256]

    results_dir = "results/Neural Field to a 2D Image"
    os.makedirs(results_dir, exist_ok=True)

    reconstructions = {}
    for f in freqs_list:
        for w in widths_list:
            recon = train_neural_field(
                image_path=image_path,
                num_freqs=f,
                hidden_dim=w,
                results_dir=results_dir,
                save_progress=(f == 10 and w == 256) 
            )
            reconstructions[(f, w)] = recon

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, f in enumerate(freqs_list):
        for j, w in enumerate(widths_list):
            axs[i, j].imshow(reconstructions[(f, w)])
            axs[i, j].set_title(f"num_freqs={f}, width={w}")
            axs[i, j].axis("off")
    plt.suptitle("Final Reconstructions (2×2 Grid)")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/final_grid.png", dpi=200)
    plt.close()

    print("Deliverables saved:")
    print(" - results/Neural Field to a 2D Image/progression_freq10_width256.png")
    print(" - results/Neural Field to a 2D Image/psnr_curve_freq10_width256.png")
    print(" - results/Neural Field to a 2D Image/final_grid.png")