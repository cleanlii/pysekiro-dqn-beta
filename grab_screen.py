# grab_screen.py

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

GAME_WIDTH   = 800    # 游戏窗口宽度
GAME_HEIGHT  = 450    # 游戏窗口高度
white_border = 31    # 游戏边框

def grab_screen(x, x_w, y, y_h):

    # 获取桌面
    hwin = win32gui.GetDesktopWindow()

    w = x_w - x
    h = y_h - y

    # 返回句柄窗口的设备环境、覆盖整个窗口，包括非客户区，标题栏，菜单，边框
    hwindc = win32gui.GetWindowDC(hwin)

    # 创建设备描述表
    srcdc = win32ui.CreateDCFromHandle(hwindc)

    # 创建一个内存设备描述表
    memdc = srcdc.CreateCompatibleDC()

    # 创建位图对象
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, w, h)
    memdc.SelectObject(bmp)
    
    # 截图至内存设备描述表
    memdc.BitBlt((0, 0), (w, h), srcdc, (x, y), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (h, w, 4)

    # 内存释放
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# 抠图
def ROI(img, x, x_w, y, y_h):
    return img[y:y_h, x:x_w]

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global vertices
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append([x, y])
        try:
            cv2.imshow("window", img)
        except NameError:
            pass
    return vertices

# 截取抠图坐标
def get_xywh(img):
    global vertices
    vertices = []

    print('Press "ESC" to quit. ')
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("window", on_EVENT_LBUTTONDOWN)
    while True:
        cv2.imshow("window", img)
        if cv2.waitKey(0)&0xFF==27:
            break
    cv2.destroyAllWindows()

    if len(vertices) != 4:
        print('Vertices number not match')
        return -1

    x = min(vertices[0][0], vertices[1][0])
    x_w = max(vertices[2][0], vertices[3][0])
    y = min(vertices[1][1], vertices[2][1])
    y_h = max(vertices[0][1], vertices[3][1])

    cv2.imshow('img', img)
    cv2.imshow('ROI(img)', ROI(img, x, x_w, y, y_h))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('ROI Coordinate:')
    print(f'\n x={x}, x_w={x_w}, y={y}, y_h={y_h}\n')

def get_game_screen():
    return grab_screen(
        x = 0,
        x_w = GAME_WIDTH,
        y = white_border,
        y_h = white_border+GAME_HEIGHT)

# 全屏
FULL_WIDTH = 1920
FULL_HEIGHT = 1080

def get_full_screen():
    return grab_screen(
        x = 0,
        x_w = FULL_WIDTH,
        y = 0,
        y_h = FULL_HEIGHT)

'''
# 获取游戏窗口参数
screen = get_full_screen()
get_xywh(screen)
'''
