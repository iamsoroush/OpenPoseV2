from os.path import normpath, abspath, dirname, realpath
PROJECT_ROOT = normpath(abspath(dirname(dirname(realpath(__file__)))))

from .aida import AIDA
from .models import OpenPoseV2
from .config import OpenPoseV2Config, HyperConfig, PoseCorrectionConfig
