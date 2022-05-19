import math
import numpy as np
import cv2
import PySimpleGUI as sg
from PIL import Image, ImageTk

edge_width = 10

def spin(label, key, val, min_val, max_val):
    return [ 
        sg.Text(label, size=(6,1)), sg.Text("", size=(6,1)), 
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(10, 1), key=key, enable_events=True )
    ]


def setPlaying(window, is_playing):
    if is_playing:
        window['-play/pause-'].update(text='Pause')
    else:
        window['-play/pause-'].update(text='Play')

    return is_playing

def show_image(image_element, img):
    # 256x256のサイズに変換する。
    img = cv2.resize(img, dsize=(256, 256))       

    if len(img.shape) == 3:
        # カラー画像の場合

        # BGRからRGBに変換する。
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # RGBからPILフォーマットへ変換する。
        image_pil = Image.fromarray(image_rgb)

    else:
        # グレースケール画像の場合

        # グレースケール画像からPILフォーマットへ変換する。
        image_pil = Image.fromarray(img)

    # PILフォーマットからImageTkフォーマットへ変換する。
    image_tk  = ImageTk.PhotoImage(image_pil) 

    # 画像を表示する。
    image_element.update(data=image_tk, size=(256,256))


def center_distance(cx, cy, contour):
    # 輪郭のモーメントを計算する。
    M = cv2.moments(contour)

    if M['m00'] == 0:
        return 10000

    # モーメントから重心のXY座標を計算す。
    bx = int(M['m10']/M['m00'])
    by = int(M['m01']/M['m00'])

    # 画像の中心から物体の重心までの変位
    dx = bx - cx
    dy = by - cy

    # 画像の中心から物体の重心までの距離を返す。
    return math.sqrt(dx * dx + dy * dy)

def bounding_rect_len(img_width, img_height, contour):
    x,y,w,h = cv2.boundingRect(contour)

    return (w + h) / (img_width + img_height)

def getContour(bin_img):

    # 二値化画像から輪郭のリストを得る。
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #  RETR_TREE RETR_CCOMP 

    # 画像の高さと幅
    img_height, img_width = bin_img.shape[:2]

    # 画像の面積
    img_area = img_width * img_height

    # 大きさが画像の10%未満の輪郭は除く。
    contours = [ c for c in contours if 0.10 < math.sqrt(cv2.contourArea(c) / img_area) ]
    if len(contours) == 0:
        # 輪郭がない場合

        print('side len')
        return [None] * 3

    # 外接矩形の辺の長さが画像の辺の長さの90%以上の輪郭は除く。
    contours = [ c for c in contours if 0.9 > bounding_rect_len(img_width, img_height, c) ]

    if len(contours) == 0:
        # 輪郭がない場合

        print('area check')
        return [None] * 3

    # 画像の中心
    cx = img_width  / 2
    cy = img_height / 2

    # 重心と画像の中心との距離が最小の輪郭のインデックス
    center_idx = np.argmin( [ center_distance(cx, cy, cont) for cont in contours ] )

    # 重心と画像の中心との距離が最小の輪郭
    contour = contours[center_idx]

    # 輪郭から0と1の二値の内部のマスク画像を作る。
    mask_img = np.zeros(bin_img.shape + (3,), dtype=np.uint8)
    cv2.drawContours(mask_img, [ contour ], -1, (1,1,1), -1)

    # 輪郭から0と1の二値の縁のマスク画像を作る。
    edge_img = np.zeros(bin_img.shape + (3,), dtype=np.uint8)
    cv2.drawContours(edge_img, [ contour ], -1, (1,1,1), edge_width)

    return contour, mask_img, edge_img
