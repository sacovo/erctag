import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from joblib import Parallel, delayed

from erctag.alvar_tags import ALVAR_TAGS


@dataclass
class Params:
    binary_threshold: int = 180

    min_tag_area: int = 100
    max_tag_area: int = 10000

    epsilon_factor: float = 0.04

    tag_padding: int = 2
    tag_gridsize: int = 5

    clip_limit: int = 2
    tile_grid_size: int = 8

    dilate_kernel_size: int = 7
    median_blur_size: int = 55


@dataclass
class Detection:
    tag_id: int
    corners: np.ndarray

    value: np.ndarray
    distance: float
    rotation: int = 0

    t: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None


ALWAYS_BLACK = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

ALWAYS_WHITE = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

black_cells_indices = np.argwhere(ALWAYS_BLACK == 0)
white_cells_indices = np.argwhere(ALWAYS_WHITE == 1)


def remove_shadows(img, dilate_kernel_size, median_blur_size):
    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(
            plane, np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        )
        bg_img = cv2.medianBlur(dilated_img, median_blur_size)
        diff_img = 255 - cv2.absdiff(plane, bg_img)

        result_planes.append(diff_img)
    shadow_rem_img = cv2.merge(result_planes)

    return shadow_rem_img


def find_threshold(black_cells, white_cells):
    """
    Finds a threshold to binarize tag values based on the given black and white cell
    criteria.

    :param black_cells: Array of cells that should be detected as black.
    :param white_cells: Array of cells that should be detected as white.
    :return: A threshold value or None if a valid threshold is not found.
    """

    max_black_value = np.max(black_cells)
    min_white_value = np.min(white_cells)

    if max_black_value < min_white_value:
        return (max_black_value + min_white_value) / 2
    else:
        return None


def validate_and_binarize_tag(tag):
    """
    Validates if the given grid values correspond to a genuine tag based on known black
    and white cells.

    Returns the binarized tag if valid, else None.

    :param grid_values: 2D array of the tag's grid values.
    :return: Binarized grid values if tag is valid, else None.
    """

    rotations = []
    for i in range(4):
        rotated_tag = np.rot90(tag, k=i)
        # Extract values for known black and white cells
        black_cells_values = [rotated_tag[i, j] for i, j in black_cells_indices]
        white_cells_values = [rotated_tag[i, j] for i, j in white_cells_indices]

        # Find threshold
        threshold = find_threshold(black_cells_values, white_cells_values)

        if threshold is None:
            continue
        rotations.append((threshold, rotated_tag, i))

    return rotations


def find_possible_tags(gray, params: Params, visualize: bool = False):
    if visualize:
        cv2.imshow("gray", gray)
        cv2.waitKey(0)

    tags_corners = []

    # Get bounding rectangle for the white region to crop the area of interest
    _, tag_area = cv2.threshold(
        gray, params.binary_threshold, 255, cv2.THRESH_BINARY_INV
    )
    if visualize:
        cv2.imshow("roi", tag_area)
    # Find contours of potential tags within the cropped ROI
    tag_contours, _ = cv2.findContours(
        tag_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if visualize:
        cv2.waitKey(0)

    for tc in tag_contours:
        perimeter = cv2.arcLength(tc, True)
        approx = cv2.approxPolyDP(tc, params.epsilon_factor * perimeter, True)

        if (
            len(approx) == 4
            and params.min_tag_area < cv2.contourArea(tc) < params.max_tag_area
        ):
            # Convert the local coordinates to global coordinates
            corners = [(pt[0][0], pt[0][1]) for pt in approx]
            tags_corners.append(corners)

    if visualize:
        cv2.destroyAllWindows()

    return tags_corners


def draw_possible_tags(img, tags_corners):
    img_copy = img.copy()

    for corners in tags_corners:
        int_corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            img_copy, [int_corners], isClosed=True, color=(0, 255, 0), thickness=2
        )

        for x, y in corners:
            cv2.circle(
                img_copy, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1
            )

    return img_copy


def order_corners(corners):
    """
    Order the four points in the following order: top-left, top-right, bottom-right,
    bottom-left.

    :param corners: List of four corner points.
    :return: Ordered list of four corner points.
    """
    sorted_corners = sorted(corners, key=lambda pt: pt[0] + pt[1])
    top_left, _, _, bottom_right = sorted_corners

    sorted_corners.remove(top_left)
    sorted_corners.remove(bottom_right)

    if sorted_corners[0][0] < sorted_corners[1][0]:
        top_right, bottom_left = sorted_corners
    else:
        bottom_left, top_right = sorted_corners

    return [top_left, top_right, bottom_right, bottom_left]


def extract_tag(img, corners, gridsize=9):
    """
    Extracts the tag defined by the given corners from the image, applies perspective
    transformation to make it a square, and then returns an array with values between 0
    and 1 representing each cell in the grid.

    :param img: Source image.
    :param corners: Four corner points of the tag.
    :param gridsize: Size of the grid inside the tag.
    :return: Array of values between 0 and 1 for each grid cell and the perspective
                transform of the tag
    """

    ordered_corners = order_corners(corners)
    # Define points for the desired perspective (a square)
    side = 250
    dst_pts = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(np.float32(ordered_corners), dst_pts)
    warped = cv2.warpPerspective(img, matrix, (side, side))

    # Normalize the values in the image to lie in range [0, 1]
    normalized = cv2.normalize(
        warped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # Sample the grid cells and compute average value for each cell
    cell_size = side // gridsize
    grid_values = np.zeros((gridsize, gridsize))

    for i in range(gridsize):
        for j in range(gridsize):
            cell = normalized[
                i * cell_size : (i + 1) * cell_size, j * cell_size : (j + 1) * cell_size
            ]
            grid_values[i, j] = np.mean(cell)

    return grid_values.T, matrix


def get_translation_vector(corners, h, K, tag_size: float):
    half_size = tag_size / 2.0
    obj_points = np.array(
        [
            [-half_size, -half_size, 0.0],
            [half_size, -half_size, 0.0],
            [half_size, half_size, 0.0],
            [-half_size, half_size, 0.0],
        ]
    )

    # Decompose the homography
    h_inv = np.linalg.inv(K) @ h
    h1 = h_inv[:, 0]
    h2 = h_inv[:, 1]
    h3 = np.cross(h1, h2)

    R_rough = np.column_stack((h1, h2, h3))
    U, _, Vt = np.linalg.svd(R_rough)
    R = U @ Vt

    t = h_inv[:, 2] / np.linalg.norm(h1)

    # Calculate apparent size in image
    center = np.mean(corners, axis=0)
    projected_corners = cv2.perspectiveTransform(
        obj_points[:, :2].reshape(-1, 1, 2), h
    ).reshape(-1, 2)
    distances = np.linalg.norm(projected_corners - center, axis=1)
    apparent_size = np.mean(distances)

    # Adjust translation using apparent size
    scale_factor = tag_size / apparent_size
    t_adjusted = t * scale_factor

    return t_adjusted, R


def visualize_tags(img, detections, font_size=1.2):
    """
    Visualizes detected tags on the original image.

    Parameters:
        img (numpy.ndarray): The original image.
        tag_data (list): List of tuples containing tag information as (tag_id, corners, confidence).

    Returns:
        numpy.ndarray: The image with visualized tags.
    """
    # Make a copy of the original image to avoid modifying it directly
    img_copy = img.copy()

    for detection in detections:
        corners, tag_id, confidence = (
            detection.corners,
            detection.tag_id,
            detection.distance,
        )
        # Draw the tag corners
        for pt in corners:
            cv2.circle(img_copy, tuple(pt), 5, (0, 255, 0), -1)

        # Draw the tag polygon
        cv2.polylines(
            img_copy,
            [np.array(corners).reshape(-1, 1, 2).astype(int)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # Write the tag ID and confidence at the tag centroid
        centroid = np.mean(corners, axis=0).astype(int)
        cv2.putText(
            img_copy,
            f"ID: {tag_id}, Dist: {confidence:.2f}",
            (centroid[0] - 150, centroid[1] - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 0, 0),
            2,
        )

    return img_copy


class TagDetector:
    def __init__(
        self,
        tag_list=ALVAR_TAGS,
        detection_params: Optional[Params] = None,
        calibration=None,
        tag_size=10,
        n_jobs=-1,
        visualize=False,
    ):
        """
        Create a new tag detector with the given configuration
        """
        if detection_params is None:
            detection_params = Params()

        self.detection_params = detection_params
        self.tag_list = np.array(tag_list)
        self.calibration = calibration
        self.n_jobs = n_jobs
        self.tag_size = tag_size
        self.visualize = visualize
        self.parallel = Parallel(n_jobs=self.n_jobs)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.detection_params.clip_limit,
            tileGridSize=(
                self.detection_params.tile_grid_size,
                self.detection_params.tile_grid_size,
            ),
        )

    def detect_tags(self, img) -> List[Detection]:
        no_shadow = remove_shadows(
            img,
            self.detection_params.dilate_kernel_size,
            self.detection_params.median_blur_size,
        )
        gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)
        self.clahe.apply(gray)

        corners = find_possible_tags(
            gray, self.detection_params, visualize=self.visualize
        )
        gridsize = self.detection_params.tag_gridsize
        padding = self.detection_params.tag_padding

        detections = self.parallel(
            delayed(extract_tag_info)(
                gray,
                corner,
                gridsize,
                padding,
                self.tag_list,
                self.calibration,
                self.tag_size,
            )
            for corner in corners
        )

        return list(filter(lambda x: x is not None, detections))


def extract_tag_info(
    gray, corner, gridsize, padding, tag_list, calibration=None, tag_size=None
):
    tag, H = extract_tag(gray, corner, gridsize + padding * 2)
    rotations = validate_and_binarize_tag(tag)

    if len(rotations) == 0:
        return None

    lowest_distance = math.inf
    detection = None

    for threshold, tag, rotation in rotations:
        tag = tag - (threshold - 0.5) / 2

        tag = tag[padding:-padding, padding:-padding].reshape(-1)
        distances = abs(tag_list - tag).sum(axis=1)

        min_idx = distances.argmin()
        distance = distances[min_idx]

        if distance <= lowest_distance:
            detection = Detection(
                tag_id=min_idx,
                value=tag,
                distance=distance,
                corners=corner,
                rotation=rotation,
            )
            lowest_distance = distance

    if detection is None:
        return None

    if calibration and tag_size:
        t, R = get_translation_vector(corner, H, calibration, tag_size)
        detection.t = t

        detection.R = R
    return detection
