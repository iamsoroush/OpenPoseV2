from os.path import normpath, abspath, dirname, realpath

from .aida import AIDA
from .models import OpenPoseV2
from .config import OpenPoseV2Config, HyperConfig, PoseCorrectionConfig


PROJECT_ROOT = normpath(abspath(dirname(dirname(realpath(__file__)))))
