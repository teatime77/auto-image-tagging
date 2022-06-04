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
from util import getContour

cap = None

playing = False
network = None

hue_shift = 10
saturation_shift = 15
value_shift = 15

classIdx = 0

S_mag =  100
V_mag =  100


class ImageClass:
    """画像のクラス(カテゴリー)
    """
    def __init__(self, name, class_dir):
        self.name = name
        self.classDir = class_dir
        self.videoPathes = []


def augment_color(img, hsv_shift):
    """画像の色を変化させてデータ拡張をする。

    Args:
        img : 入力画像
        hsv_shift : HSVの変化量

    Returns:
        データ拡張をした画像
    """
    # BGRからHSVに変える。
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    hue_shift, saturation_shift, value_shift = hsv_shift

    # HSVの各チャネルに対し
    for channel, shift in enumerate([hue_shift, saturation_shift, value_shift]):
        if shift == 0:
            continue

        # 変化量を乱数で決める。
        change = np.random.randint(-shift, shift)

        # 指定したチャネルのデータ
        data = hsv_img[:, :, channel]

        if channel == 0:
            # 色相の場合

            # int32型に変換する。
            data = data.astype(np.int32)

            # 変化後の色相が0～180の範囲に入るように色相をずらす。
            data = (data + change + 180) % 180

        else:
            # 彩度や明度の場合

            # float32型に変換する。
            data  = data.astype(np.float32)

            # 変化量は百分率(%)として作用させる。
            data  = (data * ((100 + change) / 100.0) ).clip(0, 255)

        # 変化させた値をチャネルにセットする。
        hsv_img[:, :, channel] = data.astype(np.uint8)

    # HSVからBGRに変える。
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def augment_shape(aug_img : np.ndarray, mask_img : np.ndarray, contour : np.ndarray, img_size : int) -> np.ndarray:
    """画像を回転・拡大/縮小・平行移動してデータ拡張をする。

    Args:
        aug_img : 入力画像
        mask_img : マスク画像
        contour : 輪郭
        img_size : 貼り付け先の画像のサイズ

    Returns:
        変換後の画像
    """

    # 最小外接円の中心と半径
    (cx, cy), radius = cv2.minEnclosingCircle(contour)    

    # 物体の直径
    diameter = 2 * radius

    # 最大スケール = 画像の短辺の30% ÷ 物体の直径
    max_scale = (0.3 * img_size) / diameter

    # 最小スケール = 画像の短辺の20% ÷ 物体の直径
    min_scale = (0.2 * img_size) / diameter

    # 乱数でスケールを決める。
    scale = random.uniform(min_scale, max_scale)

    # スケール変換後の半径   
    radius2 = scale * radius

    # 乱数で移動量を決める。
    margin = 1
    dx = random.uniform(radius2 - cx + margin, img_size - radius2 - cx - margin)
    dy = random.uniform(radius2 - cy + margin, img_size - radius2 - cy - margin)

    assert radius2 <= cx + dx and cx + dx <= img_size - radius2
    assert radius2 <= cy + dy and cy + dy <= img_size - radius2

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
    aug_img2  = cv2.warpAffine(aug_img , M, (img_size, img_size))
    mask_img2 = cv2.warpAffine(mask_img, M, (img_size, img_size))

    return aug_img2, mask_img2, M

def blend_image(bg_img : np.ndarray, aug_img2 : np.ndarray, mask_img2 : np.ndarray) -> np.ndarray:
    """マスク画像を使って、画像を背景画像に貼り付ける。

    Args:
        bg_img : 背景画像
        aug_img2 : 入力画像
        mask_img2 : マスク画像

    Returns:
        貼り付け後の画像
    """
    # 外接矩形
    bx, by, bw, bh = cv2.boundingRect(mask_img2)

    # 外接矩形の内部のみを抜き出す。
    aug_img2  = aug_img2[  by:(by+bh), bx:(bx+bw), : ]
    mask_img2 = mask_img2[ by:(by+bh), bx:(bx+bw) ]

    # PILフォーマットへ変換する。
    mask_pil = Image.fromarray(mask_img2)
    bg_pil   = Image.fromarray(bg_img)
    aug_pil  = Image.fromarray(aug_img2)

    # マスク画像を 収縮(erosion)する。
    mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))

    # 輪郭の周辺部分にガウシアンでぼかしを入れる。
    mask_blur = mask_pil.filter(ImageFilter.GaussianBlur(3))    

    # マスク画像を使って、データ拡張後の画像を背景画像に貼り付ける。
    bg_pil.paste(aug_pil, (bx,by), mask_blur)

    # 貼り付け後の画像をnumpyの配列に変換する。
    compo_img = np.array(bg_pil)

    return compo_img

def rotate_corners(box : list):
    """最初の頂点から2番目の頂点へ向かう辺の角度が±45°以下になるように、頂点の順番を変える。

    Args:
        box: 頂点のXY座標のリスト

    Returns:
        _type_: _description_
    """
    
    rad45 = math.radians(45)

    for i1 in range(4):
        # 次の頂点のインデックス
        i2 = (i1 + 1) % 4

        # 次の頂点までのXとYの変位
        dx = box[i2][0] - box[i1][0]
        dy = box[i2][1] - box[i1][1]

        theta = math.atan2(dy, dx)
        if abs(theta) <= rad45:
            # 次の頂点へ向かう辺の角度が±45°以下の場合

            # i1が最初の頂点になるようにする。
            return box[i1:] + box[:i1]

    return None

def resize_bg_img(bg_img : np.ndarray, img_size : int) -> np.ndarray:
    """背景画像を指定したサイズの正方形にする。

    Args:
        bg_img : 背景画像
        img_size : 正方形の辺の長さ

    Returns:
        指定したサイズの正方形の背景画像
    """
    # 背景画像の高さと幅
    h, w = bg_img.shape[:2]

    # 高さと幅の小さい方
    size = min(h, w)

    # 正方形の画像の開始位置
    y = (h - size) // 2
    x = (w - size) // 2

    # 背景画像を正方形にする。
    bg_img = bg_img[ y:(y+size), x:(x+size), :]

    # 指定したサイズにリサイズする。
    bg_img = cv2.resize(bg_img, dsize=(img_size, img_size))

    return bg_img

def make_train_data(frame, bg_img, img_size, v_min, hsv_shift):
    # グレー画像
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # 二値画像
    bin_img = 255 - cv2.inRange(gray_img, v_min, 255)

    # 二値化画像から輪郭とマスク画像を得る。
    msg, contour, mask_img = getContour(bin_img)
    if msg != '' or bg_img is None:
        return [bin_img] + [None] * 6

    # 画像の色を変化させてデータ拡張をする。
    aug_img = augment_color(frame, hsv_shift)

    # 回転を考慮した外接矩形を得る。
    rect = cv2.minAreaRect(contour)

    # 外接矩形の頂点
    box = cv2.boxPoints(rect)

    aug_img2, mask_img2, M  = augment_shape(aug_img, mask_img, contour, img_size)

    # 背景画像を指定したサイズにする。
    bg_img = resize_bg_img(bg_img, img_size)             

    compo_img = blend_image(bg_img, aug_img2, mask_img2)

    # 頂点に変換行列をかける。
    corners2 = [ np.dot(M, np.array(p + [1])).tolist() for p in box.tolist() ]

    # 最初の頂点から2番目の頂点へ向かう辺の角度が±45°以下になるように、頂点の順番を変える。
    corners2 = rotate_corners(corners2)
    if corners2 is None:
        print('slope is None')
        
        return [bin_img] + [None] * 6

    # バウンディングボックスと回転角を得る。
    bounding_box = _corners2rotatedbbox(corners2)

    x, y, w, h, theta = bounding_box
    if not (0 <= x and x + w <= img_size and 0 <= y and y + h <= img_size):
        print(f'x:{x} x+w:{x+w} y:{y} y+h:{y+h} img-size:{img_size}')
        return [bin_img] + [None] * 6

    return bin_img, mask_img, compo_img, aug_img, box, corners2, bounding_box

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
    parser.add_argument('-net','--network', type=str, help='odtk or yolov5', default='odtk')
    parser.add_argument('-dtsz', '--data_size', type=int, help='data size', default=1000)
    parser.add_argument('-imsz', '--img_size', type=int, help='image size', default=720)
    parser.add_argument('-v', '--v_min', type=int, help='Value min', default=130)
    parser.add_argument('-hsv', '--hsv_shift', type=int, nargs=3, help='color shift', default=(10,15,15))

    args = parser.parse_args(sys.argv[1:])

    # 動画ファイルのフォルダのパス
    video_dir = args.input.replace('\\', '/')

    # 背景画像ファイルのフォルダのパス
    bg_img_dir = args.bg.replace('\\', '/')

    # 出力先フォルダのパス
    output_dir = args.output.replace('\\', '/')

    network_name = args.network.lower()

    return video_dir, bg_img_dir, output_dir, network_name, args.data_size, args.img_size, args.v_min, args.hsv_shift

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

def get_total_frame_count(image_class):
    cnt = 0
    for video_path in image_class.videoPathes:

        cap = get_video_capture(video_path)    

        if not cap.isOpened():
            print("動画再生エラー")
            sys.exit()

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cnt += n

        cap.release()

    return cnt

def make_training_data(image_classes, bg_img_paths, network, data_size, img_size, v_min, hsv_shift):

    # 背景画像ファイルのインデックス
    bg_img_idx = 0

    for class_idx, image_class in enumerate(image_classes):
        total_frame_cnt = get_total_frame_count(image_class)

        class_data_cnt = 0

        video_idx = 0
        cap = init_cap(image_class, video_idx)
        while class_data_cnt < data_size:

            ret, frame = cap.read()
            if ret:
                # 画像が取得できた場合

                if data_size < total_frame_cnt and data_size / total_frame_cnt < np.random.rand():
                    continue

                # 動画の現在位置
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

                # 背景画像ファイルを読む。
                bg_img = cv2.imread(bg_img_paths[bg_img_idx])
                bg_img_idx = (bg_img_idx + 1) % len(bg_img_paths)

                bin_img, mask_img, compo_img, aug_img, box, corners2, bounding_box = make_train_data(frame, bg_img, img_size, v_min, hsv_shift)

                if mask_img is not None:
                    network.add_image(class_idx, video_idx, pos, compo_img, corners2, bounding_box)

                    class_data_cnt += 1

                    yield

                    if data_size <= class_data_cnt:
                        # 現在のクラスのテータ数が指定値に達した場合

                        break

            else:

                video_idx = (video_idx + 1) % len(image_class.videoPathes)
                cap = init_cap(image_class, video_idx)

    network.save()

if __name__ == '__main__':
    video_dir, bg_img_dir, output_dir, network_name, data_size, img_size, v_min, hsv_shift = parse()
    hue_shift, saturation_shift, value_shift = hsv_shift

    # OpenCVのビルド情報を表示する。
    # print(cv2.getBuildInformation())

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

    hsv_shift = (hue_shift, saturation_shift, value_shift)
    iterator = make_training_data(image_classes, bg_img_paths, network, data_size, img_size, v_min, hsv_shift)
    for _ in tqdm(iterator, total=len(image_classes) * data_size):
        pass
