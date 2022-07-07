from encoders.encoders import EncodersManager
from encoders.vit import VitEncoder
from encoders.resnet import ResnetEncoder
try:
    from encoders.open_clip import OpenClip
except:
    pass