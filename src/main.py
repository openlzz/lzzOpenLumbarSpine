import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO("best.pt")

# 更新模型的类别名称
model.model.names.update({
    0: 'l1_l2_正常', 1: 'l1_l2_moderate', 2: '1_l2_严重狭窄',
    3: 'scs_l2_l3_正常', 4: 'scs_l2_l3_moderate', 5: 'scs_l2_l3_严重狭窄',
    6: 'scs_l3_l4_正常', 7: 'scs_l3_l4_moderate', 8: 'scs_l3_l4_严重狭窄',
    9: 'scs_l4_l5_正常', 10: 'scs_l4_l5_moderate', 11: 'scs_l4_l5_严重狭窄',
    12: 'scs_l5_s1_正常', 13: 'scs_l5_s1_moderate', 14: 'scs_l5_s1_严重狭窄'
})

def run(dicom_path):
    # 读取 DICOM 文件
    ds = pydicom.dcmread(dicom_path)
    # 将 DICOM 图像数据转换为 numpy 数组
    img_array = ds.pixel_array
    
    # 将图像数据转换为 OpenCV 格式
    img = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 使用 YOLO 模型进行检测
    results = model(img)
     
    # 展示检测结果
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        result.show()  # display to screen

    return results
if __name__=='__main__':
        
    # 调用函数，传入 DICOM 文件的路径
    run('/path/to/your/dicom/file.dcm')
