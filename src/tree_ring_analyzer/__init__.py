__version__ = "0.1.0"
__release__ = "dev" # "dev" or "stable"

import re

ORIGINAL_UNIT = "µm"
TIFF_REGEX = re.compile(r"(.+)\.tiff?", re.IGNORECASE)
INPUT_SIZE = (256, 256, 1)
FILTER_NUM = [16, 24, 40, 80, 960]
N_LABELS = 1