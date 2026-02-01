import numpy as np
from part2_1_rays import pixel_to_ray


# ------------------------------------------------------------
# Load LEGO Data
# ------------------------------------------------------------
data = np.load("data/lego_200x200.npz")

images_train = data["images_train"].astype(np.float32) / 255.0
c2ws_train   = data["c2ws_train"].astype(np.float32)
focal        = float(data["focal"])

H, W = images_train.shape[1:3]

K = np.array([
    [focal, 0, W / 2],
    [0, focal, H / 2],
    [0, 0, 1]
], dtype=np.float32)

print(f"Loaded training data: {images_train.shape[0]} images ({H}x{W})")
print("Focal length:", focal)
print("K:\n", K)


# ------------------------------------------------------------
# FAST Vectorized Ray Sampling (NO LOOPS!)
# ------------------------------------------------------------
def sample_rays_from_images(images, c2ws, K, n_rays=1024):
    """
    Returns:
        rays_o : (n_rays,3)
        rays_d : (n_rays,3)
        rgbs   : (n_rays,3)
    """

    n_images, H, W, _ = images.shape

    # random image + random pixels
    img_inds = np.random.randint(0, n_images, (n_rays,))
    u = np.random.randint(0, W, (n_rays,))
    v = np.random.randint(0, H, (n_rays,))

    # Pixel centers
    uv = np.stack([u + 0.5, v + 0.5], axis=-1).astype(np.float32)

    # Ray origins & directions
    rays_o = np.zeros((n_rays, 3), dtype=np.float32)
    rays_d = np.zeros((n_rays, 3), dtype=np.float32)

    # **Vectorized by image index grouping**
    for img_id in np.unique(img_inds):
        mask = np.where(img_inds == img_id)[0]

        uv_sel = uv[mask]               # (M,2)
        c2w_sel = c2ws[img_id]          # (4,4)

        o, d = pixel_to_ray(K, c2w_sel, uv_sel)
        rays_o[mask] = o
        rays_d[mask] = d

    # RGB values
    rgbs = images[img_inds, v, u]   # (n_rays,3)

    return rays_o, rays_d, rgbs.astype(np.float32)


# ------------------------------------------------------------
# Sample points along rays
# ------------------------------------------------------------
def sample_along_rays(ray_o, ray_d, n_samples=64, near=2.0, far=6.0, perturb=True):
    """
    ray_o : (N,3)
    ray_d : (N,3)
    Returns:
        points: (N, n_samples, 3)
        t_vals: (N, n_samples)
    """
    N = ray_o.shape[0]

    # Uniform sampling
    t_vals = np.linspace(near, far, n_samples, dtype=np.float32)

    if perturb:
        # midpoints for stratified sampling
        mids = 0.5 * (t_vals[1:] + t_vals[:-1])
        lower = np.concatenate([[t_vals[0]], mids])
        upper = np.concatenate([mids, [t_vals[-1]]])

        # random sampling per ray
        t_rand = np.random.rand(N, n_samples).astype(np.float32)
        t_vals = lower + (upper - lower) * t_rand  # (N,n_samples)
    else:
        t_vals = np.broadcast_to(t_vals, (N, n_samples))

    # Compute 3D samples:   o + d * t
    points = ray_o[:, None, :] + ray_d[:, None, :] * t_vals[..., None]
    return points.astype(np.float32), t_vals.astype(np.float32)


# ------------------------------------------------------------
# Debug / Test
# ------------------------------------------------------------
if __name__ == "__main__":
    rays_o, rays_d, rgbs = sample_rays_from_images(
        images_train, c2ws_train, K, n_rays=8
    )

    points, t_vals = sample_along_rays(rays_o, rays_d, n_samples=8)

    print("ray_o:", rays_o.shape)
    print("ray_d:", rays_d.shape)
    print("rgbs:", rgbs.shape)
    print("points:", points.shape)
    print("t_vals range:", t_vals.min(), "→", t_vals.max())

    print("\nExample Ray 0:")
    print("Origin:", rays_o[0])
    print("Direction:", rays_d[0])
    print("RGB:", rgbs[0])
    print("First 3 points:\n", points[0, :3])