from typing import List

import numpy as np
from fundus_odmac_toolkit.models.segmentation import segment


def segment_od_mask(imgs: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    ods = []
    macs = []
    for img in imgs:
        result = segment(img, **kwargs).argmax(0)

        ods.append((result == 1).cpu().numpy())
        macs.append((result == 2).cpu().numpy())

    return ods, macs
