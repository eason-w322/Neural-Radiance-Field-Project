import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# FAST Transform: Camera → World
# ------------------------------------------------------------
def transform(c2w: np.ndarray, x_c: np.ndarray) -> np.ndarray:
    """
    c2w : (4,4)
    x_c : (N,3)
    returns x_w : (N,3)
    """
    R = c2w[:3, :3]        # (3,3)
    t = c2w[:3, 3]         # (3,)
    return x_c @ R.T + t   # (N,3)


# ------------------------------------------------------------
# Pixel (u,v) → camera point at depth
# ------------------------------------------------------------
def pixel_to_camera(K: np.ndarray, uv: np.ndarray, depth=1.0) -> np.ndarray:
    """
    uv : (N,2)
    returns (N,3) camera coords at Z=depth
    """

    u = uv[:, 0] + 0.5    # pixel center fix
    v = uv[:, 1] + 0.5

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = np.full_like(x, depth, dtype=np.float32)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ------------------------------------------------------------
# Pixel → World Ray (origin, direction)
# ------------------------------------------------------------
def pixel_to_ray(K: np.ndarray, c2w: np.ndarray, uv: np.ndarray):
    """
    Returns:
        ray_o: (N,3)
        ray_d: (N,3)
    """
    uv = uv.astype(np.float32)

    # camera origin (1,3)
    o = c2w[:3, 3].reshape(1, 3).astype(np.float32)

    # camera point at z=1
    x_c = pixel_to_camera(K, uv, depth=1.0)

    # transform to world
    x_w = transform(c2w, x_c)

    # ray directions
    d = x_w - o
    d /= (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)

    # repeat origin
    o = np.repeat(o, uv.shape[0], axis=0)

    return o.astype(np.float32), d.astype(np.float32)


# ------------------------------------------------------------
# Transform → Verify
# ------------------------------------------------------------
def verify_transform():
    R = np.eye(3)
    t = np.array([0.1, 0.2, 0.3])
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t

    x_c = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x_w = transform(c2w, x_c)

    c2w_inv = np.linalg.inv(c2w)
    x_c_back = transform(c2w_inv, x_w)

    print("Original x_c:", x_c)
    print("World x_w:", x_w)
    print("Back-projected x_c:", x_c_back)
    assert np.allclose(x_c, x_c_back, atol=1e-6)
    print("transform() verified ✓\n")


# ------------------------------------------------------------
# Central Ray Example
# ------------------------------------------------------------
def verify_ray_example():
    fx = fy = 100.0
    cx = cy = 50.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)

    c2w = np.eye(4, dtype=np.float32)

    uv_center = np.array([[50.0, 50.0]], dtype=np.float32)

    o, d = pixel_to_ray(K, c2w, uv_center)
    print("Ray origin:", o)
    print("Ray direction:", d)
    print("Expected direction ≈ [0,0,1]\n")


# ------------------------------------------------------------
# LEGO Example Visualization
# ------------------------------------------------------------
def example_with_lego():
    data = np.load("data/lego_200x200.npz")

    images_train = data["images_train"].astype(np.float32) / 255.0
    c2ws_train = data["c2ws_train"].astype(np.float32)
    focal = float(data["focal"])

    H, W = images_train.shape[1:3]
    K = np.array([[focal, 0, W / 2],
                  [0, focal, H / 2],
                  [0, 0, 1]], dtype=np.float32)

    img = images_train[1]
    c2w = c2ws_train[1]

    # small grid around center
    u = np.linspace(W / 2 - 2, W / 2 + 2, 5)
    v = np.linspace(H / 2 - 2, H / 2 + 2, 5)
    uu, vv = np.meshgrid(u, v)
    uv = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    o, d = pixel_to_ray(K, c2w, uv)

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.quiver(uv[:, 0], uv[:, 1],
               d[:, 0], -d[:, 1],
               color='red', angles='xy',
               scale_units='xy', scale=0.05)
    plt.title("Ray directions (image plane projection)")
    plt.axis("off")
    plt.show()


# ------------------------------------------------------------
# Run Tests
# ------------------------------------------------------------
if __name__ == "__main__":
    verify_transform()
    verify_ray_example()
    example_with_lego()