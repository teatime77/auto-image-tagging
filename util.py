import numpy as np
import cv2
import PySimpleGUI as sg
from PIL import Image, ImageTk

Next_Sibling, Previous_Sibling, First_Child, Parent = (0, 1, 2, 3)

max_ratio, max_rx, max_ry = 0, 0, 0

def spin(label, key, val, min_val, max_val):
    return [ 
        sg.Text(label, size=(6,1)), sg.Text("", size=(6,1)), 
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(10, 1), key=key, enable_events=True )
    ]


def show_image(image_element, img):
    img = cv2.resize(img, dsize=(256, 256))       

    if len(img.shape) == 3:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        # print('image_rgb:type', type(image_rgb), image_rgb.shape, image_rgb.dtype)
        image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
    else:
        image_pil = Image.fromarray(img) # RGBからPILフォーマットへ変換

    image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換

    image_element.update(data=image_tk, size=(256,256))


def isObject(shape, contour):
    global max_ratio, max_rx, max_ry

    # 二値画像の幅と高さ
    width, height = shape[:2]

    # 二値画像の面積
    img_area = width * height

    # 輪郭の面積
    area = cv2.contourArea(contour)
    ratio = 100 * np.sqrt(area) / np.sqrt(img_area)

    if max_ratio < ratio:
        max_ratio = ratio

    if ratio < 40:
        return False

    # 輪郭のモーメントを計算する。
    M = cv2.moments(contour)

    # モーメントから重心のXY座標を計算す。
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    rx = int(100 * cx / width)
    ry = int(100 * cy / height)

    if max_ratio == ratio:
        max_rx, max_ry = rx, ry

    return 35 <= rx and rx <= 65 and 35 <= ry and ry <= 65

def contour_children(contours, hierarchy, idx, children):
    # 最初の子
    i = hierarchy[0][idx][First_Child]
    while i != -1:
        c = contours[i]
        children.append(c)

        j = hierarchy[0][idx][First_Child]
        if j != -1:
            contour_children(contours, hierarchy, j, children)

        # 次の兄弟
        i = hierarchy[0][i][Next_Sibling]


def getContour(bin_img):
    global max_ratio, max_rx, max_ry

    # 二値化画像から輪郭のリストを得る。
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # RETR_EXTERNAL  RETR_CCOMP 

    assert(len(hierarchy.shape) == 3 and hierarchy.shape[0] == 1 and hierarchy.shape[2] == 4)
    assert(len(contours) == hierarchy.shape[1])

    max_ratio, max_rx, max_ry = 0, 0, 0

    for idx, _ in enumerate(contours):
        if hierarchy[0][idx][Parent] == -1:
            # トップレベルの場合

            contour = contours[idx]
            if isObject(bin_img.shape, contour):

                contour_family = [ contour ]
                contour_children(contours, hierarchy, idx, contour_family)


                # 輪郭から0と1の二値の内部のマスク画像を作る。
                mask_img = np.zeros(bin_img.shape + (3,), dtype=np.uint8)
                cv2.drawContours(mask_img, contour_family, -1, (1,1,1), -1)

                # 輪郭から0と1の二値の縁のマスク画像を作る。
                edge_img = np.zeros(bin_img.shape + (3,), dtype=np.uint8)
                cv2.drawContours(edge_img, contour_family, -1, (1,1,1), 5)



                return contour, contour_family, mask_img, edge_img

    print(f'ratio : {int(max_ratio)} < 40   35 < rx:{int(max_rx)} < 65 and 35 < ry:{int(max_ry)} < 65')

    return [None] * 4
