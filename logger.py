import logging

nnlogger: logging.Logger = logging.getLogger("pynnlib")

# Enable this to debug imports
import sys
# nnlogger.addHandler(logging.StreamHandler(sys.stdout))
nnlogger.setLevel("DEBUG")
