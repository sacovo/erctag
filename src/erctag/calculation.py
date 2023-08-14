import cv2
import numpy as np


def compute_angle_and_distance(corners, tag_size, camera_matrix, distortion_coeffs):
    corners = np.array(corners, dtype=np.float32)
    corners = cv2.undistortPoints(
        corners.reshape(-1, 1, 2),
        camera_matrix,
        distortion_coeffs,
        None,
        camera_matrix,
    )

    z = compute_distance(corners, tag_size, camera_matrix)

    theta_x, theta_y = compute_angle(corners, camera_matrix)

    return z, theta_x, theta_y


def compute_distance(tag_corners, real_tag_size, camera_matrix):
    # Real-world coordinates of the tag corners assuming Z=0 for the tag plane
    half_size = real_tag_size / 2.0
    obj_points = np.array(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
        ],
        dtype=np.float32,
    )

    # Compute the homography matrix from real-world to image coordinates
    h, _ = cv2.findHomography(obj_points, tag_corners)

    # Reproject the real-world corners using homography
    reprojected_corners = cv2.perspectiveTransform(
        obj_points.reshape(-1, 1, 2), h
    ).reshape(-1, 2)

    # Compute the apparent size based on the longest side
    side_lengths = [
        np.linalg.norm(reprojected_corners[i] - reprojected_corners[(i + 1) % 4])
        for i in range(4)
    ]
    apparent_size = np.mean(side_lengths)

    # Focal length (taking an average of fx and fy)
    f = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0

    # Calculate distance based on triangle similarity
    Z = (f * real_tag_size) / apparent_size

    return Z


def compute_angle(corners, camera_matrix):
    # 1. Get the principal point from the camera matrix
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 2. Get the tag's center
    tag_center = np.mean(corners, axis=0).ravel()
    x_center, y_center = tag_center

    # 3. Compute the vector from the principal point to the tag's center
    dx = x_center - cx
    dy = y_center - cy

    # 4. Calculate the angle
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    theta_x = np.arctan2(dx, fx)
    theta_y = np.arctan2(dy, fy)

    return np.degrees(theta_x), np.degrees(theta_y)
