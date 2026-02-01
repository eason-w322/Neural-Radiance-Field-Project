
import numpy as np
import time
import viser
from part2_1_rays import transform, pixel_to_camera, pixel_to_ray
from part2_2_sampling import sample_along_rays


class RaysData:
    def __init__(self, images, K, c2ws):
        self.images = images
        self.K = K
        self.c2ws = c2ws

        n_images, H, W, _ = images.shape
        print(f"Loaded {n_images} images ({H}x{W})")

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        self.uvs = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32) + 0.5

        rays_o_all, rays_d_all, rgbs_all = [], [], []

        for img_idx, (img, c2w) in enumerate(zip(images, c2ws)):
            ray_o, ray_d = pixel_to_ray(K, c2w, self.uvs)
            ray_o = np.broadcast_to(ray_o, ray_d.shape)
            rays_o_all.append(ray_o)
            rays_d_all.append(ray_d)
            rgbs_all.append(img.reshape(-1, 3))

        self.rays_o = np.concatenate(rays_o_all, axis=0)
        self.rays_d = np.concatenate(rays_d_all, axis=0)
        self.pixels = np.concatenate(rgbs_all, axis=0)

        print(f"Total rays: {self.rays_o.shape[0]}")
        print("RaysData initialization complete!\n")

    def sample_rays(self, n_rays=1024):
        idx = np.random.randint(0, self.pixels.shape[0], size=n_rays)
        return self.rays_o[idx], self.rays_d[idx], self.pixels[idx]


if __name__ == "__main__":
    data = np.load("data/lego_200x200.npz")
    images_train = data["images_train"] / 255.0
    c2ws_train = data["c2ws_train"]
    focal = float(data["focal"])

    H, W = images_train.shape[1:3]
    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1],
    ])
    print("Intrinsic matrix K:\n", K)

    dataset = RaysData(images_train, K, c2ws_train)

    uvs_start, uvs_end = 0, 40_000
    sample_uvs = dataset.uvs[uvs_start:uvs_end]
    assert np.allclose(
        images_train[0, sample_uvs[:, 1].astype(int), sample_uvs[:, 0].astype(int)],
        dataset.pixels[uvs_start:uvs_end],
    ), "❌ UV-to-pixel mapping mismatch!"
    print("✅ UV mapping verified!\n")

    n_rays = 200
    n_images = len(c2ws_train)
    rays_o, rays_d, pixels = [], [], []

    for _ in range(n_rays):
        img_idx = np.random.randint(0, n_images)
        offset = img_idx * (H * W)
        pixel_idx = np.random.randint(0, H * W)
        idx = offset + pixel_idx
        rays_o.append(dataset.rays_o[idx])
        rays_d.append(dataset.rays_d[idx])
        pixels.append(dataset.pixels[idx])

    rays_o = np.stack(rays_o)
    rays_d = np.stack(rays_d)
    points, _ = sample_along_rays(rays_o, rays_d, n_samples=32, perturb=True)


    server = viser.ViserServer(share=True)

    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image,
        )

    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        positions = np.stack((o, o + d * 6.0))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=positions,
        )

    server.scene.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.03,
    )

    print("\n(viser) Visualization ready! Click the generated share URL above ⬆️")
    while True:
        time.sleep(0.1)
    