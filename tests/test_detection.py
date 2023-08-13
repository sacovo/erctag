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
