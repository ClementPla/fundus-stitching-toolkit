from enum import Enum, auto


class ReferenceChoice(Enum):
    OD = auto()
    MACULA = auto()
    MAX_VESSELS = auto()


class Descriptors(Enum):
    ORB = auto()
    SIFT = auto()
    SURF = auto()
    BRIEF = auto()
    BRISK = auto()
    FREAK = auto()
    AKAZE = auto()
    TOPO_BASED = auto()
    DINOV2 = auto()
    CNN = auto()


class Keypoints(Enum):
    VESSELS_CROSSING = auto()
    SIFT = auto()
    SURF = auto()
    ORB = auto()
    MODEL_KLARGEST = auto()
