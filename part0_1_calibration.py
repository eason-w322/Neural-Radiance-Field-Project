import cv2
import numpy as np
import glob
import os

calib_dir = "data/calib2_images"
output_file = "data/camera_calib.npz"

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

image_paths = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))
print(f"Found {len(image_paths)} images in {calib_dir}")

tag_size = 0.057        # each tag width
spacing_x = 0.085       # horizontal spacing 
spacing_y = 0.071    # vertical spacing 

grid_positions = {
    0: (0, 0),
    1: (spacing_x, 0),
    2: (0, spacing_y),
    3: (spacing_x, spacing_y),
    4: (0, 2 * spacing_y),
    5: (spacing_x, 2 * spacing_y),
}

objpoints = []  # list of 3D points per image
imgpoints = []  # list of 2D points per image


for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is None or len(corners) == 0:
        print(f"[Skip] No tags detected in {os.path.basename(path)}")
        continue

    objp_all, imgp_all = [], []

    for c, marker_id in zip(corners, ids.flatten()):
        if marker_id not in grid_positions:
            continue  

        offset_x, offset_y = grid_positions[marker_id]
        objp_marker = np.array([
            [offset_x, offset_y, 0],
            [offset_x + tag_size, offset_y, 0],
            [offset_x + tag_size, offset_y + tag_size, 0],
            [offset_x, offset_y + tag_size, 0],
        ], dtype=np.float32)

        objp_all.append(objp_marker)
        imgp_all.append(c.reshape(-1, 2))

    if len(objp_all) > 0:
        objpoints.append(np.concatenate(objp_all))
        imgpoints.append(np.concatenate(imgp_all))
    else:
        print(f"[Skip] No valid grid markers found in {os.path.basename(path)}")

print(f"Processed {len(objpoints)} valid calibration images.")

if len(objpoints) > 0:
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print(K.dtype)

    print(f"\nRMS reprojection error: {ret:.4f}")
    print("Camera matrix (K):\n", K)
    print("Distortion coefficients:\n", dist.ravel())

    errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        projected = projected.reshape(-1, 2).astype(np.float32)
        observed = imgpoints[i].astype(np.float32)
        err = cv2.norm(observed, projected, cv2.NORM_L2) / len(projected)
        errors.append(err)
    print(f"Mean per-image reprojection error: {np.mean(errors):.4f} px")


    h, w = gray.shape[:2]
    np.savez(output_file, K=K, dist=dist, img_size=(h, w))
    print(f"{h},{w}")
    print(f"Calibration results saved to {output_file}")
else:
    print("No valid calibration images found. Check tag visibility and re-run.")