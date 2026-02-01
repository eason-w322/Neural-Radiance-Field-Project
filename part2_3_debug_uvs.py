import numpy as np
import time
import viser

from part2_1_rays import pixel_to_ray
from part2_2_sampling import sample_along_rays


if __name__ == "__main__":
    # Load data
    data = np.load("data/lego_200x200.npz")
    images = data["images_train"] / 255.0
    c2ws = data["c2ws_train"]
    focal = float(data["focal"])

    H, W = images.shape[1:3]
    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])

    # -------------------------------
    # 🔥 Choose camera 0 ONLY
    # -------------------------------
    cam_id = 0
    img = images[cam_id]
    c2w = c2ws[cam_id]

    # Make UV grid for THIS camera only
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uvs = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32) + 0.5

    # Sample 200 random pixels
    idx = np.random.randint(0, H*W, 200)
    selected_uvs = uvs[idx]

    # Compute rays only from this one camera
    rays_o, rays_d = pixel_to_ray(K, c2w, selected_uvs)

    # Sample 3D points along these rays
    points, _ = sample_along_rays(rays_o, rays_d, n_samples=32, perturb=True)

    # -------------------------------
    # 🔥 Viser visualization
    # -------------------------------
    server = viser.ViserServer(share=True)

    # Show ALL frustums (but no rays from them)
    for i, (im, pose) in enumerate(zip(images, c2ws)):
        server.scene.add_camera_frustum(
            f"/camera/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.12,
            wxyz=viser.transforms.SO3.from_matrix(pose[:3, :3]).wxyz,
            position=pose[:3, 3],
            image=im,
        )

    # Only show rays from THIS camera
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        server.scene.add_spline_catmull_rom(
            f"/ray/{i}",
            positions=np.stack([o, o + d * 6], axis=0)
        )

    # Add sample points
    server.scene.add_point_cloud(
        "/samples",
        points=points.reshape(-1, 3),
        colors=np.zeros_like(points).reshape(-1, 3),
        point_size=0.03,
    )

    print("Viser running — only ONE camera should show rays.")
    while True:
        time.sleep(0.1)