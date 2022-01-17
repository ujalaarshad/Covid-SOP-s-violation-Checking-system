import cv2
import numpy as np
import time
from scipy.spatial import distance_matrix



def heatmap(points, top):
    k = 21
    gauss = cv2.getGaussianKernel(k, np.sqrt(64))
    gauss = gauss * gauss.T
    gauss = gauss / gauss[k//2, k//2]
    spark = cv2.cvtColor(cv2.applyColorMap((gauss * 255).astype(np.uint8), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB).astype(np.float32)/255

    heat = np.zeros(top.shape).astype(np.float32)

    for p in points:
        heat[p[1] - k // 2: 1 + p[1] + k // 2, p[0] - k // 2: 1 + p[0] + k // 2, :] += spark

    heat = heat / (np.max(heat, axis=(0, 1)) + 0.0001)
    gray = cv2.cvtColor(heat, cv2.COLOR_RGB2GRAY)
    mask = np.where(gray > 0.2, 1, 0).astype(np.float32)
    mask_3 = np.ones((top.shape[0], top.shape[1], 3)) * (1-mask)[:, :, None]
    mask_4 = heat * mask[:, :, None]
    new_top = (top * mask_3) + mask_4
    return new_top




            
            
       
