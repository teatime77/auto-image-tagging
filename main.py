import os
import sys
import math
import random
import argparse
import glob
import cv2
import numpy as np
from odtk import _corners2rotatedbbox, ODTK
from yolo_v5 import YOLOv5
from util import spin, show_image, getContour, edge_width, setPlaying

data_size = 1000
playing = False
network = None

hue_shift = 10
saturation_shift = 15
value_shift = 15

classIdx = 0
imageClasses = []

# 背景画像ファイルのインデックス
bgImgIdx = 0

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


def make_train_data(frame, bg_img):
    # グレー画像
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # 二値画像
    bin_img = 255 - cv2.inRange(gray_img, V_lo, 255)

    # 輪郭とマスク画像とエッジ画像を得る。
    contour, mask_img, edge_img = getContour(bin_img)
    if contour is None:
        return

    aug_img = augment_color(frame)

    # 元画像にマスクをかける。
    clip_img = aug_img * mask_img

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
    dst_img2 = cv2.warpAffine(clip_img, M, (width, height))
    aug_img2 = cv2.warpAffine(aug_img, M, (width, height))
    mask_img2 = cv2.warpAffine(mask_img, M, (width, height))
    edge_img2 = cv2.warpAffine(edge_img, M, (width, height))

    # 背景画像を元画像と同じサイズにする。
    bg_img = cv2.resize(bg_img, dsize=aug_img.shape[:2])                    

    # 内部のマスクを使って、背景画像と元画像を合成する。
    compo_img = np.where(mask_img2 == 0, bg_img, aug_img2)

    # 背景と元画像を7対3の割合で合成する。
    blend_img = cv2.addWeighted(bg_img, 0.7, aug_img2, 0.3, 0.0)

    # 縁の部分をブレンドした色で置き換える。
    compo_img = np.where(edge_img2 == 0, compo_img, blend_img)

    # 頂点に変換行列をかける。
    corners2 = [ np.dot(M, np.array(p + [1])).tolist() for p in box.tolist() ]

    # 最初の頂点から2番目の頂点へ向かう辺の角度が±45°以下になるように、頂点の順番を変える。
    corners2 = rotate_corners(corners2)
    if corners2 is None:
        print('slope is None')
        
        return

    # 座標変換後の外接矩形を描く。
    cv2.drawContours(dst_img2, [ np.int0(corners2)  ], 0, (0,255,0), 2)

    # バウンディングボックスと回転角を得る。
    bounding_box = _corners2rotatedbbox(corners2)
    x, y, w, h, theta = bounding_box

    # バウンディングボックスを描く。
    cv2.rectangle(dst_img2, (int(x),int(y)), (int(x+w),int(y+h)), (0,0,255), 3)

    # バウンディングボックスの左上の頂点の位置に円を描く。
    cv2.circle(dst_img2, (int(x), int(y)), 10, (255,255,255), -1)

    return frame, gray_img, bin_img, clip_img, dst_img2, compo_img, corners2, bounding_box

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
    parser.add_argument('-net','--network', type=str, help='odtk or yolov5')

    args = parser.parse_args(sys.argv[1:])

    # 動画ファイルのフォルダのパス
    video_dir = args.input.replace('\\', '/')

    # 背景画像ファイルのフォルダのパス
    bg_img_dir = args.bg.replace('\\', '/')

    # 出力先フォルダのパス
    output_dir = args.output.replace('\\', '/')

    network_name = args.network.lower()

    return video_dir, bg_img_dir, output_dir, network_name

class CaptureIterator(object):
    def __init__(self, *numbers):
        self._numbers = numbers
        self._i = 0

    def __iter__(self):
        # __next__()はselfが実装してるのでそのままselfを返す
        return self

    def __next__(self):
        ret, frame = cap.read()
        if ret:
            # 画像が取得できた場合

            # 動画の現在位置
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

            # 背景画像ファイルを読む。
            bg_img = cv2.imread(bgImgPaths[bgImgIdx])
            bgImgIdx = (bgImgIdx + 1) % len(bgImgPaths)

            frame, gray_img, bin_img, clip_img, dst_img2, compo_img, corners2, bounding_box = make_train_data(frame, bg_img)

            if network is not None:

                network.add_image(class_idx, video_idx, pos, compo_img, corners2, bounding_box)

                class_data_cnt += 1

                if class_data_cnt % 10 == 0:
                    print(image_class.name, class_data_cnt)

                if data_size <= class_data_cnt:
                    # 現在のクラスのテータ数が指定値に達した場合

                    # キャプチャー オブジェクトを解放する。
                    cap.release()

                    break

        else:

            # キャプチャー オブジェクトを解放する。
            cap.release()

            if network is None:
                video_idx += 1

                if video_idx < len(image_class.videoPathes):
                    self.init_cap()

                else:

                    class_idx += 1


            else:
                video_idx = (video_idx + 1) % len(image_class.videoPathes)


            if self._i == len(self._numbers):
                raise StopIteration()
            value = self._numbers[self._i]
            self._i += 1
            return value

    def init_cap(self):
        video_path = image_class.videoPathes[video_idx]

        # 動画のキャプチャー オブジェクト
        cap = cv2.VideoCapture(video_path)    

        if not cap.isOpened():
            print("動画再生エラー")
            sys.exit()


def make_network_data():
    if network_name == 'odtk':
        network = ODTK(output_dir, imageClasses)
    elif network_name == 'yolov5':
        network = YOLOv5(output_dir, imageClasses)

    else:
        assert(False)

    print(cv2.getBuildInformation())

    for class_idx, image_class in enumerate(imageClasses):
        class_data_cnt = 0

        video_idx = 0
        while class_data_cnt < data_size:


            while True:

                yield

    network.save()

if __name__ == '__main__':
    video_dir, bg_img_dir, output_dir, network_name = parse()

    print(cv2.getBuildInformation())

    # 出力先フォルダを作る。
    os.makedirs(output_dir, exist_ok=True)

    # 背景画像ファイルのパス
    bgImgPaths = [ x for x in glob.glob(f'{bg_img_dir}/*') if os.path.splitext(x)[1] in [ '.jpg', '.png' ] ]

    imageClasses = make_image_classes(video_dir)

    for _ in make_network_data():
        pass
