import cv2
from PIL import Image
import numpy as np

img = Image.open(
    'c:/selfie.jpg'
)

img = img.resize((512, 512))
img.save(
    'c:/data/selfie.jpg'
)