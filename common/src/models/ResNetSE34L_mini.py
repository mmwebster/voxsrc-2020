from models.ResNetSE34L import ResNetSE
from models.ResNetBlocks import *

def ResNetSE34L_mini(nOut=256, **kwargs):
    model = ResNetSE(SEBasicBlock, layers=[1, 2, 3, 1],
                     num_filters=[8, 16, 32, 64], nOut=nOut, **kwargs)
    return model
