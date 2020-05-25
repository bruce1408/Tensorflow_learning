import os
import math
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
"""
图片转字节使用两个函数 tobytes 或者是 tostring函数都是可以的
"""
path = "starry_night_dd3.jpg"
image = Image.open(path)
image = image.resize((224, 224), Image.ANTIALIAS)
image = np.array(image)
print(image.shape)
img_raw = image.tobytes()
print(img_raw)
img = image.tostring()
print(img)