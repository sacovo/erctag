import os

import click
import cv2
import numpy as np

from erctag import TagDetector
from erctag.detection import Params, visualize_tags


@click.group()
def cli():
    pass


@cli.command()
@click.argument("folder")
@click.option("--binary-threshold", type=int, default=180)
@click.option("--min-tag-area", type=int, default=100)
@click.option("--max-tag-area", type=int, default=10_000)
@click.option("--clip-limit", type=int, default=2)
@click.option("--tile-grid-size", type=int, default=8)
@click.option("--dilate-kernel-size", type=int, default=7)
@click.option("--median-blur-size", type=int, default=55)
@click.option("--calibration", type=click.Path())
@click.option("--tag-size", type=int, default=10)
@click.option("--visualize/--no-visualize", default=False)
@click.option("--run-name", default=None)
def detect(
    folder, calibration=None, tag_size=10, visualize=True, run_name=None, **kwargs
):
    params = Params(**kwargs)

    if calibration:
        calibration = np.loadtxt(calibration)
    else:
        calibration = None

    detector = TagDetector(
        detection_params=params,
        calibration=calibration,
        tag_size=tag_size,
        visualize=visualize,
    )

    if os.path.isdir(folder):
        paths = sorted(os.listdir(folder))
    else:
        paths = [os.path.basename(folder)]
        folder = os.path.dirname(folder)

    for path in paths:
        img = cv2.imread(os.path.join(folder, path))
        detection = detector.detect_tags(img)

        out = visualize_tags(img, detection, font_size=1.2)
        cv2.putText(
            out,
            f"{run_name if run_name else 'Path'}: {os.path.basename(path)}",
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            2,
        )

        cv2.imshow("output", out)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
