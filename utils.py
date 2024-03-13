import pydicom
import math
import cv2
import numpy as np
from PIL import Image
import os
import shutil

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt





def img_preprocessing(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    Pixel_Spacing = ds.PixelSpacing[1]
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(40,20))
    test_clahe = clahe.apply(scaled_image)
    test_clahe_RGB = cv2.cvtColor(test_clahe, cv2.COLOR_GRAY2RGB)
    return test_clahe_RGB, scaled_image,Pixel_Spacing 


def get_foot(point, line):
    start_x, start_y = line[0], line[1]
    end_x, end_y = line[2], line[3]
    pa_x, pa_y = point
 
    p_foot = [0, 0]
    if line[0] == line[3]:
        p_foot[0] = line[0]
        p_foot[1] = point[1]
        return p_foot
    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot[0] = int((b * b * pa_x - a * b * pa_y - a * c) / (a * a + b * b))
    p_foot[1] = int((a * a * pa_y - a * b * pa_x - b * c) / (a * a + b * b))
    return p_foot


def distance(x1, y1, x2, y2,PixelSpacing):
    return math.sqrt((x2*PixelSpacing - x1*PixelSpacing) ** 2 + (y2*PixelSpacing - y1*PixelSpacing) ** 2)   

def create_or_delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' has been deleted and recreated.")    
    else:
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' has been created.")


def bland_altman_plot(data1, data2, color='white',edgecolor='green',dpi=600,Y_axis_offset_up=0.15,Y_axis_offset_down=0.65,save_path= r'C:\Users\lijie\OneDrive\adenoid\figures\test.eps'):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)  # data1与data2的均值
    diff = data1 - data2  # data1与data2的差值
    md = np.mean(diff)  # data1与data2差值的平均值
    sd = np.std(diff, axis=0)  # data1与data2差值的标准差
    
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(7.5, 5), dpi= dpi)
    ax = plt.axes()
    ax.set_facecolor('white')
    ax.set_xmargin(0.16)

    plt.axhline(md, color='purple', linestyle='-.', lw=2, zorder=1)
    plt.axhline(md + 1.96 * sd, color=sns.color_palette()[3], linestyle='--', lw=2, zorder=1)

    print('差值的均值为：%.3f (%.3f ~ %.3f)' % (md, md - 1.96 * sd, md + 1.96 * sd))
    plt.axhline(md - 1.96 * sd, color=sns.color_palette()[3], linestyle='--', lw=2)
    plt.scatter(mean, diff, color='white',edgecolor='green', marker="o", s=60, alpha=1, zorder=2)
    plt.ylim(np.min(diff)-sd, np.max(diff)+sd)
    
    ax.spines['top'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)

    # 区间上界
    plt.text(max(mean) + 0.008, md + 1.96 * sd + Y_axis_offset_up, '1.96 SD', fontsize=14)
    plt.text(max(mean) + 0.070, md + 1.96 * sd - Y_axis_offset_down, '%.3f' % (md + 1.96 * sd), fontsize=14)
    # 均值线
    plt.text(max(mean) + 0.070, md + Y_axis_offset_up, 'Mean', fontsize=14)
    plt.text(max(mean) + 0.070, md - Y_axis_offset_down, '%.3f' % md, fontsize=14)
    # 区间下界
    plt.text(max(mean) - 0.045, md - 1.96 * sd + Y_axis_offset_up, '-1.96 SD', fontsize=14)
    plt.text(max(mean) - 0.010, md - 1.96 * sd - Y_axis_offset_down, '%.3f' % (md - 1.96 * sd), fontsize=14)

    plt.xlabel("Mean value between system and reference standard", fontsize=12)  
    plt.ylabel("Difference between system and reference standard", fontsize=12)
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.tick_params(width=1.5, labelsize=14)  
    plt.savefig(save_path, dpi=dpi)
    plt.show()