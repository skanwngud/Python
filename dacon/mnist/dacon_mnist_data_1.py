import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

for i in range(50000):
    filepath=('../data/dacon/data/dirty_mnist/%05d.png'%i)
    dataset=Image.open(filepath)
