import CT_regression_tools
import numpy as np
from pathlib import Path

model = CT_regression_tools.VGG_16()
pth = Path(__file__).parent
model.save(pth / 'vgg.h5')