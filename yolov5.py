import os
import sys
import random
import json
import shutil
import argparse

from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser(description='YOLOv5 training data')
    parser.add_argument('-i','--input', type=str, help='path to videos', default=os.getcwd())
    parser.add_argument('-o','--output', type=str, help='path to outpu', default=os.getcwd())

    args = parser.parse_args(sys.argv[1:])

    if args.output == '.':
        args.output = os.getcwd()

    elif args.output[0] != '/' and not ':' in args.output:
        args.output = f'{os.getcwd()}/{args.output}'

    args.input, args.output = (args.input.replace('\\', '/'), args.output.replace('\\', '/'))

    print(f'input : {args.input}')
    print(f'output: {args.output}')

    return args.input, args.output

if __name__ == '__main__':
    input_dir, outpu_dir = parse()

    with open(f'{input_dir}/train.json') as f:
        obj =json.load(f)

    train_class_names = [x["name"] for x in obj["categories"]]

    os.makedirs(f'{outpu_dir}/datasets/images/train', exist_ok=True)
    os.makedirs(f'{outpu_dir}/datasets/images/val', exist_ok=True)
    os.makedirs(f'{outpu_dir}/datasets/labels/train', exist_ok=True)
    os.makedirs(f'{outpu_dir}/datasets/labels/val', exist_ok=True)

    with open(f'{outpu_dir}/datasets/data.yaml', 'w') as f:
        # データフォルダのパス
        f.write(f'path: {outpu_dir}/datasets\n')

        # トレーニング用の画像ファイルの相対パス
        f.write(f'train: images/train\n')

        # バリデーション用の画像ファイルの相対パス
        f.write(f'val: images/val\n')

        # クラスの数
        f.write(f'nc: {len(train_class_names)}\n')

        # クラスの名前のリスト
        f.write(f'names: {train_class_names}\n')

    random.shuffle(obj['images'])

    for idx, img_inf in enumerate( tqdm(obj['images'], total=len(obj['images'])) ):
        # 画像id
        img_id = img_inf['id']

        # 画像idに対応するアノテーション
        ann_list = [x for x in obj['annotations'] if x["image_id"] == img_id]

        assert len(ann_list) == 1
        ann = ann_list[0]

        # 95%をトレーニングに使い、5%をバリデーションに使う。
        if idx < len(obj['images']) * 95 / 100:

            # トレーニングの場合

            images_dir = f'{outpu_dir}/datasets/images/train'
            labels_dir = f'{outpu_dir}/datasets/labels/train'

        else:
            # バリデーションの場合

            images_dir = f'{outpu_dir}/datasets/images/val'
            labels_dir = f'{outpu_dir}/datasets/labels/val'

        # コピー元の画像ファイルのパス
        image_path = f'{input_dir}/img/{img_inf["file_name"]}'

        # 画像ファイルを画像フォルダにコピーする。
        shutil.copy(image_path, images_dir)

        # 画像の高さと幅
        image_height = float(img_inf["height"])
        image_width  = float(img_inf["width"])

        # 物体の位置とサイズ
        x, y, w, h, theta = ann['bbox']
        
        # 物体の中心の位置
        x_center = (x + 0.5 * w) / image_width
        y_center = (y + 0.5 * h) / image_height

        # 物体のサイズ
        width    = w / image_width
        height   = h / image_height

        # ラベルファイル名
        label_file_name = img_inf["file_name"].replace('.jpg', '.txt')

        # ラベルファイルのパス
        label_file_path = f'{labels_dir}/{label_file_name}'

        # ラベルファイルに書く。
        with open(label_file_path, 'w') as f:
            f.write(f'{ann["category_id"]} {x_center} {y_center} {width} {height}\n')