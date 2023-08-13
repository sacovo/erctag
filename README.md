# ERC Tag Detection

This library provides detection capabilites for ALVAR Tags, that are used by the ERC.


## Usage

If you only care about the position of the tags in the image, you can just create a
detector like so:

```python

from erctag import TagDetector, visualize_tags
import cv2

detector = TagDetector()

img = cv2.imread(...)
tags = detector.detect_tags(img)

out = visualize_tags(img, tags, font_size=1.2)
cv2.imshow("Tags", out)
cv2.waitKey()

```

You can follow the detection process by specifying `visualize=True`.

### Detection Parameters

You can configure the detection using the parameters in erctag.detection.Params:

```python
from erctag import Params
params = Params(
    binary_threshold=120, # For finding the tag
    median_blur_size=55, # For the shadow removal algorithm
)
```

For all parameters look inside the definition.


## Translation and Orientation

If you have the camera calibration and the tag size, you can also calculate the relative
position of tags to the camera, just provide the matrix and the size to the detector.

```python
matrix = np.array(
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    }
)
tag_size = 22


# Make sure focal length and tag size units match
# Detections now have t and R for translation and Rotation
detector = TagDetector(calibration=matrix, tag_size=tag_size)

```
