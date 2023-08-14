import os
from pathlib import Path

import cv2
import numpy as np

from erctag.detection import TagDetector

BASE_PATH = Path(os.path.dirname(__file__))

TEST_DATA = [
    {
        "path": BASE_PATH / "data/examples/img_000506_1691417462194.jpg",
        "tags": [2],
    },
    {
        "path": BASE_PATH / "data/examples/img_000703_1691417715553.jpg",
        "tags": [3],
    },
]


def test_some_images():
    detector = TagDetector()

    for case in TEST_DATA:
        img = cv2.imread(str(case["path"]))
        tags = detector.detect_tags(img)

        assert len(tags) >= len(case["tags"])

        tag_ids = [tag.tag_id for tag in tags]

        for tag in case["tags"]:
            assert tag in tag_ids


def test_rotated_images():
    detector = TagDetector()

    for case in TEST_DATA:
        img = cv2.imread(str(case["path"]))

        for i in range(3):
            rotated = np.rot90(img, -i)
            tags = detector.detect_tags(rotated)

            assert len(tags) >= len(case["tags"])

            tag_ids = [tag.tag_id for tag in tags]

            for tag in case["tags"]:
                assert tag in tag_ids

            for tag in tags:
                assert tag.rotation == i


def test_with_calib():
    mtx = np.array(
        [
            [1500, 0, 670.0],
            [0, 1700, 370],
            [0, 0, 1],
        ]
    ).astype(np.float32)

    detector = TagDetector(calibration=mtx, tag_size=20, n_jobs=1)

    for case in TEST_DATA:
        img = cv2.imread(str(case["path"]))
        tags = detector.detect_tags(img)

        assert len(tags) >= len(case["tags"])

        tag_ids = [tag.tag_id for tag in tags]

        for tag in case["tags"]:
            assert tag in tag_ids


def test_with_calib_and_dist():
    mtx = np.array(
        [
            [1500, 0, 670.0],
            [0, 1700, 370],
            [0, 0, 1],
        ]
    ).astype(np.float32)
    dst_coef = np.array(
        [[1.3628861, -5.59947733, -0.01674108, -0.22750355, 10.2141071]]
    )

    detector = TagDetector(calibration=mtx, distortion=dst_coef, tag_size=20, n_jobs=1)

    for case in TEST_DATA:
        img = cv2.imread(str(case["path"]))
        tags = detector.detect_tags(img)

        assert len(tags) >= len(case["tags"])

        tag_ids = [tag.tag_id for tag in tags]

        for tag in case["tags"]:
            assert tag in tag_ids
