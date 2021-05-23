# get_status.py

# 检测思路1：兴趣区域->灰度图->检测像素值->快速
# 检测思路2：ROI抠图坐标->边缘检测->量化为数字->适用性广

import cv2
import numpy as np
from grab_screen import get_game_screen, get_xywh, ROI

# screen = get_game_screen()
# get_xywh(screen)

# 注：架势值截取右半部分图像即可，部分敌人无架势影响，则架势值无法读取

# 识别自身血量
def get_self_HP(img):
    img = ROI(img, x=48, x_w=307, y=406, y_h=410)
    canny = cv2.Canny(cv2.GaussianBlur(img,(3,3),0), 0, 100)
    value = canny.argmax(axis=-1)
    return np.median(value)

# 识别自身架势值
def get_self_posture(img):
    img = ROI(img, x=402, x_w=491, y=388, y_h=390)
    canny = cv2.Canny(cv2.GaussianBlur(img,(3,3),0), 0, 100)
    value = canny.argmax(axis=-1)
    return np.median(value)

# 识别boss血量
def get_boss_HP(img):
    img = ROI(img, x=48, x_w=215, y=40, y_h=45)
    canny = cv2.Canny(cv2.GaussianBlur(img,(3,3),0), 0, 100)
    value = canny.argmax(axis=-1)
    return np.median(value)

# 识别boss架势值
def get_boss_posture(img):
    img = ROI(img, x=402, x_w=554, y=27, y_h=31)
    canny = cv2.Canny(cv2.GaussianBlur(img,(3,3),0), 0, 100)
    value = canny.argmax(axis=-1)
    return np.median(value)

# 封装以便统一调用
def get_game_status(img):
    return get_self_HP(img), get_self_posture(img), get_boss_HP(img), get_boss_posture(img)

# # 测试
# while True:
#     screen = get_game_screen()
#     status = get_game_status(screen)
#     print(f'\r {str(status):<30}', end='')
#     cv2.waitKey(1)

