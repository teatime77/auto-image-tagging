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

        self.output_dir = output_dir


        images_train = f'{output_dir}/datasets/images/train'
        images_val   = f'{output_dir}/datasets/images/val'  
        labels_train = f'{output_dir}/datasets/labels/train'
        labels_val   = f'{output_dir}/datasets/labels/val'  

        for dir_path in [ images_train, images_val, labels_train, labels_val ]:
            if os.path.isdir(dir_path):
                # フォルダがすでにある場合

                # フォルダを削除する。
                shutil.rmtree(dir_path)

            # フォルダを作る。
            os.makedirs(dir_path, exist_ok=True)

        with open(f'yolo_v5.yaml', 'w') as f:
            # データフォルダのパス
            f.write(f'path: ./datasets\n')

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

    def add_image(self, class_idx, video_idx, pos, compo_img, corners2, bbox):
        # 画像IDをカウントアップする。
        self.image_id += 1

        # 画像ファイルとラベルファイルの名前
        name = f'{class_idx}-{video_idx}-{pos}-{self.image_id}'

        # 9/10をトレーニングに使い、1/10をバリデーションに使う。
        is_train = ( random.uniform(0.0, 1.0) < 0.9 )

        train_val = 'train' if is_train else 'val'

        # 画像ファイルのパス
        image_path = f'{self.output_dir}/datasets/images/{train_val}/{name}.jpg'

        # ラベルのパス
        label_path = f'{self.output_dir}/datasets/labels/{train_val}/{name}.txt'

        # 画像ファイルに書く。
        cv2.imwrite(image_path, compo_img)

        # 画像の高さと幅
        image_height, image_width = compo_img.shape[:2]

        # 物体の位置とサイズ
        x, y, w, h, theta = bbox
        
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