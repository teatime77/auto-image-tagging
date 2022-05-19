import imp
import os
import json
import cv2
import random
import shutil

class YOLOv5:
    def __init__(self, output_dir, image_classes):
        self.image_id = 0
        self.image_classes = image_classes

        self.datasets_dir = f'{output_dir}/datasets'

        if os.path.isdir(self.datasets_dir):
            # フォルダがすでにある場合

            # フォルダを削除する。
            shutil.rmtree(self.datasets_dir)


        images_train = f'{self.datasets_dir}/images/train'
        images_val   = f'{self.datasets_dir}/images/val'  
        labels_train = f'{self.datasets_dir}/labels/train'
        labels_val   = f'{self.datasets_dir}/labels/val'  

        for dir_path in [ images_train, images_val, labels_train, labels_val ]:

            # フォルダを作る。
            os.makedirs(dir_path, exist_ok=True)

        with open(f'{self.datasets_dir}/yolo_v5.yaml', 'w') as f:
            # データフォルダのパス
            f.write(f'path: .\n')

            # トレーニング用の画像ファイルの相対パス
            f.write(f'train: images/train\n')

            # バリデーション用の画像ファイルの相対パス
            f.write(f'val: images/val\n')

            # クラスの数
            f.write(f'nc: {len(image_classes)}\n')

            # クラスの名前のリスト
            f.write(f'names: {[x.name for x in image_classes]}\n')

    def images_cnt(self):
        return self.image_id

    def add_image(self, class_idx, video_idx, pos, compo_img, corners2, bounding_box):
        # 画像IDをカウントアップする。
        self.image_id += 1

        # 画像ファイルとラベルファイルの名前
        name = f'{class_idx}-{video_idx}-{pos}-{self.image_id}'

        # 19/20をトレーニングに使い、1/20をバリデーションに使う。
        is_train = ( random.uniform(0.0, 1.0) < 0.95 )

        train_val = 'train' if is_train else 'val'

        # 画像ファイルのパス
        image_path = f'{self.datasets_dir}/images/{train_val}/{name}.jpg'

        # ラベルのパス
        label_path = f'{self.datasets_dir}/labels/{train_val}/{name}.txt'

        # 画像ファイルに書く。
        cv2.imwrite(image_path, compo_img)

        # 画像の高さと幅
        image_height, image_width = compo_img.shape[:2]

        # 物体の位置とサイズ
        x, y, w, h, theta = bounding_box
        
        # 物体の中心の位置
        x_center = (x + 0.5 * w) / float(image_width)
        y_center = (y + 0.5 * h) / float(image_height)

        # 物体のサイズ
        width    = w / float(image_width)
        height   = h / float(image_height)

        # ラベルファイルに書く。
        with open(label_path, 'w') as f:
            f.write(f'{class_idx} {x_center} {y_center} {width} {height}\n')

    def save(self):
        pass