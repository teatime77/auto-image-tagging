import os
import sys
import math
import random
import argparse
import glob
import cv2
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
from odtk import _corners2rotatedbbox, ODTK
from yolo_v5 import YOLOv5
from util import getContour, edge_width

cap = None

playing = False
network = None

hue_shift = 10
saturation_shift = 15
value_shift = 15

classIdx = 0

V_lo = 250

S_mag =  100
V_mag =  100


class ImageClass:
    """画像のクラス(カテゴリー)
    """
    def __init__(self, name, class_dir):
        self.name = name
        self.classDir = class_dir
        self.videoPathes = []

class CaptureIterator(object):
    def __init__(self):
        pass

    def __iter__(self):
    #     return self

    # def __next__(self):
        for i in range(10):
            yield i

        raise StopIteration()


def augment_color(img):
    # BGRからHSVに変える。
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    # HSVの各チャネルに対し
    for channel, shift in enumerate([hue_shift, saturation_shift, value_shift]):
        if shift == 0:
            continue

        # 変化量を乱数で決める。
        change = np.random.randint(-shift, shift)

        if channel == 0:
            # 色相の場合

            data = hsv_img[:, :, channel].astype(np.int32)

            # 変化後の色相が0～180の範囲に入るように色相をずらす。
            data = (data + change + 180) % 180

        else:
            # 彩度や明度の場合

            data  = hsv_img[:, :, channel].astype(np.float32)

            # 変化量は百分率(%)として作用させる。
            data  = (data * ((100 + change) / 100.0) ).clip(0, 255)

        # 変化させた値をチャネルにセットする。
        hsv_img[:, :, channel] = data.astype(np.uint8)

    # HSVからBGRに変える。
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        

def rotate_corners(box):
    rad45 = math.radians(45)

    for i1 in range(4):
        i2 = (i1 + 1) % 4

        dx = box[i2][0] - box[i1][0]
        dy = box[i2][1] - box[i1][1]

        theta = math.atan2(dy, dx)
        if abs(theta) <= rad45:
            return box[i1:] + box[:i1]

    return None

def resize_bg_img(bg_img, img_size):
    h, w = bg_img.shape[:2]

    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2

    y2 = y1 + size
    x2 = x1 + size
    bg_img = bg_img[y1:y2, x1:x2, :]
    bg_img = cv2.resize(bg_img, dsize=(img_size, img_size))

    return bg_img

def make_train_data(frame, bg_img, img_size, V_lo):
    # グレー画像
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # 二値画像
    bin_img = 255 - cv2.inRange(gray_img, V_lo, 255)

    # 輪郭とマスク画像とエッジ画像を得る。
    contour, mask_img, edge_img = getContour(bin_img)
    if contour is None:
        return [frame, gray_img, bin_img] + [None] * 4

    aug_img = augment_color(frame)

    # 元画像にマスクをかける。
    # clip_img = aug_img * mask_img
    clip_img = aug_img.copy()

    cv2.drawContours(clip_img, [ contour ], -1, (255,0,0), edge_width)

    # 回転を考慮した外接矩形を得る。
    rect = cv2.minAreaRect(contour)

    # 外接矩形の頂点
    box = cv2.boxPoints(rect)

    # 外接矩形を描く。
    cv2.drawContours(clip_img, [ np.int0(box) ], 0, (0,255,0), 2)

    # 最小外接円の中心と半径
    (cx, cy), radius = cv2.minEnclosingCircle(contour)    

    # 最小外接円を描く。
    cv2.circle(clip_img, (int(cx), int(cy)), int(radius), (255,255,255), 1)

    if bg_img is None:
        return frame, gray_img, bin_img, clip_img, None, None, None

    # 画像の高さと幅
    height, width = aug_img.shape[:2]

    # 画像の短辺の長さ
    min_size = min(height, width)

    # 物体の直径
    diameter = 2 * radius

    # 最大スケール = 画像の短辺の30% ÷ 物体の直径
    max_scale = (0.3 * min_size) / diameter

    # 最小スケール = 画像の短辺の20% ÷ 物体の直径
    min_scale = (0.2 * min_size) / diameter

    # 乱数でスケールを決める。
    scale = random.uniform(min_scale, max_scale)

    # スケール変換後の半径   
    radius2 = scale * radius

    # 乱数で移動量を決める。
    margin = 1
    dx = random.uniform(radius2 - cx + margin, width - radius2 - cx - margin)
    dy = random.uniform(radius2 - cy + margin, height - radius2 - cy - margin)

    assert radius2 <= cx + dx and cx + dx <= width - radius2
    assert radius2 <= cy + dy and cy + dy <= height - radius2

    # 乱数で回転量を決める。
    angle = random.uniform(-180, 180)

    # 回転とスケール
    m1 = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    m1 = np.concatenate((m1, np.array([[0.0, 0.0, 1.0]])))

    # 平行移動
    m2 = np.array([
        [ 1, 0, dx], 
        [ 0, 1, dy], 
        [ 0, 0,  1]
    ], dtype=np.float32)

    m3 = np.dot(m2, m1)
    M = m3[:2,:]

    # 画像に変換行列を作用させる。
    aug_img2  = cv2.warpAffine(aug_img, M, (width, height))
    mask_img2 = cv2.warpAffine(mask_img, M, (width, height))

    bx, by, bw, bh = cv2.boundingRect(mask_img2)

    aug_img2  = aug_img2[  by:(by+bh), bx:(bx+bw), : ]
    mask_img2 = mask_img2[ by:(by+bh), bx:(bx+bw) ]

    # 背景画像を元画像と同じサイズにする。
    bg_img = resize_bg_img(bg_img, img_size)             

    # PILフォーマットへ変換する。
    mask_pil = Image.fromarray(mask_img2)
    bg_pil   = Image.fromarray(bg_img)
    aug_pil  = Image.fromarray(aug_img2)

    for _ in range(1):
        mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))

    mask_blur = mask_pil.filter(ImageFilter.GaussianBlur(3))    

    bg_pil.paste(aug_pil, (bx,by), mask_blur)

    compo_img = np.array(bg_pil)

    # 頂点に変換行列をかける。
    corners2 = [ np.dot(M, np.array(p + [1])).tolist() for p in box.tolist() ]

    # 最初の頂点から2番目の頂点へ向かう辺の角度が±45°以下になるように、頂点の順番を変える。
    corners2 = rotate_corners(corners2)
    if corners2 is None:
        print('slope is None')
        
        return [frame, gray_img, bin_img] + [None] * 4

    # バウンディングボックスと回転角を得る。
    bounding_box = _corners2rotatedbbox(corners2)

    return frame, gray_img, bin_img, clip_img, compo_img, corners2, bounding_box

def make_image_classes(video_dir):
    image_classes = []

    for class_dir in glob.glob(f'{video_dir}/*'):
        category_name = os.path.basename(class_dir)

        img_class = ImageClass(category_name, class_dir)
        image_classes.append(img_class)

        # クラスのフォルダ内の動画ファイルに対し
        for video_path in glob.glob(f'{class_dir}/*'):

            video_path_str = str(video_path).replace('\\', '/')

            img_class.videoPathes.append(video_path_str)

    return image_classes

def parse():
    parser = argparse.ArgumentParser(description='Auto Image Tag')
    parser.add_argument('-i','--input', type=str, help='path to videos')
    parser.add_argument('-bg', type=str, help='path to background images')
    parser.add_argument('-o','--output', type=str, help='path to outpu')
    parser.add_argument('-net','--network', type=str, help='odtk or yolov5', default='')
    parser.add_argument('-dtsz', '--data_size', type=int, help='data size', default=1000)
    parser.add_argument('-imsz', '--img_size', type=int, help='image size', default=720)

    args = parser.parse_args(sys.argv[1:])

    # 動画ファイルのフォルダのパス
    video_dir = args.input.replace('\\', '/')

    # 背景画像ファイルのフォルダのパス
    bg_img_dir = args.bg.replace('\\', '/')

    # 出力先フォルダのパス
    output_dir = args.output.replace('\\', '/')

    network_name = args.network.lower()

    return video_dir, bg_img_dir, output_dir, network_name, args.data_size, args.img_size

def get_video_capture(video_path):
    global cap

    if cap is not None:
        cap.release()

    cap = cv2.VideoCapture(video_path)    

    return cap

def init_cap(image_class, video_idx):
    video_path = image_class.videoPathes[video_idx]

    # 動画のキャプチャー オブジェクト
    cap = get_video_capture(video_path)    

    if not cap.isOpened():
        print("動画再生エラー")
        sys.exit()

    return cap


def make_training_data(image_classes, bg_img_paths, network, data_size, img_size):

    # 背景画像ファイルのインデックス
    bg_img_idx = 0

    for class_idx, image_class in enumerate(image_classes):
        class_data_cnt = 0

        video_idx = 0
        cap = init_cap(image_class, video_idx)
        while class_data_cnt < data_size:

            ret, frame = cap.read()
            if ret:
                # 画像が取得できた場合

                # 動画の現在位置
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

                # 背景画像ファイルを読む。
                bg_img = cv2.imread(bg_img_paths[bg_img_idx])
                bg_img_idx = (bg_img_idx + 1) % len(bg_img_paths)

                frame, gray_img, bin_img, clip_img, compo_img, corners2, bounding_box = make_train_data(frame, bg_img, img_size, V_lo)

                yield

                if clip_img is not None:
                    network.add_image(class_idx, video_idx, pos, compo_img, corners2, bounding_box)

                    class_data_cnt += 1

                    if data_size <= class_data_cnt:
                        # 現在のクラスのテータ数が指定値に達した場合

                        break

            else:

                video_idx = (video_idx + 1) % len(image_class.videoPathes)
                cap = init_cap(image_class, video_idx)

    network.save()

if __name__ == '__main__':
    video_dir, bg_img_dir, output_dir, network_name, data_size, img_size = parse()

    print(cv2.getBuildInformation())

    # 出力先フォルダを作る。
    os.makedirs(output_dir, exist_ok=True)

    # 背景画像ファイルのパス
    bg_img_paths = [ x for x in glob.glob(f'{bg_img_dir}/*') if os.path.splitext(x)[1] in [ '.jpg', '.png' ] ]

    image_classes = make_image_classes(video_dir)

    if network_name == 'odtk':
        network = ODTK(output_dir, image_classes)
    elif network_name == 'yolov5':
        network = YOLOv5(output_dir, image_classes)

    else:
        assert(False)

    iterator = make_training_data(image_classes, bg_img_paths, network, data_size, img_size)
    for _ in tqdm(iterator, total=len(image_classes) * data_size):
        pass
