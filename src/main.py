import os
import pandas as pd
import numpy as np
import pydicom
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2
import glob
from ultralytics import YOLO

model = YOLO("/kaggle/input/spineyolotst0/lsdc_yolo11/train/weights/best.pt")

model.model.names.update({0: 'l1_l2_正常', 1: 'l1_l2_moderate', 2: '1_l2_严重狭窄', 3: 'scs_l2_l3_正常', 4: 'scs_l2_l3_moderate', 5: 'scs_l2_l3_严重狭窄', 6: 'scs_l3_l4_正常', 7: 'scs_l3_l4_moderate', 8: 'scs_l3_l4_严重狭窄', 9: 'scs_l4_l5_正常', 10: 'scs_l4_l5_moderate', 11: 'scs_l4_l5_严重狭窄', 12: 'scs_l5_s1_正常', 13: 'scs_l5_s1_moderate', 14: 'scs_l5_s1_严重狭窄'})



a=glob.glob('/kaggle/input/spineyolotst0/data_fold0/images/val/*')
len(a)

print(model.names)

for i in a[10:20]:
    results=model(i)
    results[0].show()

results[0].show()
results=model(a[10])


for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs

    result.show(font_size=0.1,line_width=1,labels=True,pil=False)  # display to screen 默认用cv画图，文字是现线宽的1/3
