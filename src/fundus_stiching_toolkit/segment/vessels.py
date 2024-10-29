from typing import List

import fundus_vessels_toolkit.segment as fst_segment
import numpy as np


def segment_vessels(imgs: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    output = []

    for img in imgs:
        result = fst_segment.segment_vessels(img, **kwargs)

        output.append(result)

    return output
