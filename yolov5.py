import os
import sys
import random
import json
import shutil
import argparse

from tqdm import tqdm

def parse():
    """コマンドライン引数を解析する。

    Returns: 入力フォルダと出力フォルダのパス
    """
    parser = argparse.ArgumentParser(description='COCO形式からYOLOv5形式に学習データを変換する。')
    parser.add_argument('-i','--input', type=str, help='入力フォルダのパス ( COCO形式の学習データ)', default=os.getcwd())
    parser.add_argument('-o','--output', type=str, help='出力フォルダのパス (YOLOv5形式の学習データ)', default=os.getcwd())

    args = parser.parse_args(sys.argv[1:])

    # data.yamlの中の"path"を絶対パスで指定するために、出力フォルダを絶対パスにする。
    if args.output == '.':
        # 出力がカレントフォルダの場合

        # 絶対パスにする。
        args.output = os.getcwd()

    elif args.output[0] != '/' and not ':' in args.output:
        # 出力が相対パスの場合

        # 絶対パスにする。
        args.output = f'{os.getcwd()}/{args.output}'

    # パスの区切り文字をUNIX形式に統一する。
    args.input, args.output = (args.input.replace('\\', '/'), args.output.replace('\\', '/'))

    print(f'input : {args.input}')
    print(f'output: {args.output}')

    return args.input, args.output

if __name__ == '__main__':

    # 入力フォルダと出力フォルダのパスをコマンドライン引数から得る。
    input_dir, outpu_dir = parse()

    # COCO形式のアノテーションファイルを読む。
    with open(f'{input_dir}/train.json') as f:
        obj =json.load(f)

    # クラスの名前のリスト
    class_names = [x["name"] for x in obj["categories"]]

    # トレーニング用とバリデーション用の画像ファイルのフォルダを作る。
    os.makedirs(f'{outpu_dir}/datasets/images/train', exist_ok=True)
    os.makedirs(f'{outpu_dir}/datasets/images/val', exist_ok=True)

    # トレーニング用とバリデーション用のラベルファイルのフォルダを作る。
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
        f.write(f'nc: {len(class_names)}\n')

        # クラスの名前のリスト
        f.write(f'names: {class_names}\n')

    # 画像ファイルのリストをシャッフルする。
    random.shuffle(obj['images'])

    # すべての画像ファイルに対して
    for idx, img_inf in enumerate( tqdm(obj['images'], total=len(obj['images'])) ):
        # 画像id
        img_id = img_inf['id']

        # 画像idに対応するアノテーションのリスト
        ann_list = [x for x in obj['annotations'] if x["image_id"] == img_id]

        # 画像idに対応するアノテーションは1個のはず
        assert len(ann_list) == 1
        ann = ann_list[0]

        # 95%をトレーニングに使い、5%をバリデーションに使う。
        if idx < len(obj['images']) * 95 / 100:
            # トレーニングの場合

            # YOLOv5用の画像ファイルとラベルファイルのパス
            images_dir = f'{outpu_dir}/datasets/images/train'
            labels_dir = f'{outpu_dir}/datasets/labels/train'

        else:
            # バリデーションの場合

            # YOLOv5用の画像ファイルとラベルファイルのパス
            images_dir = f'{outpu_dir}/datasets/images/val'
            labels_dir = f'{outpu_dir}/datasets/labels/val'

        # コピー元の画像ファイルのパス
        src_image_path = f'{input_dir}/img/{img_inf["file_name"]}'

        # コピー元の画像ファイルをYOLOv5用の画像フォルダにコピーする。
        shutil.copy(src_image_path, images_dir)

        # 画像の高さと幅
        image_height = float(img_inf["height"])
        image_width  = float(img_inf["width"])

        # 物体の位置とサイズと回転。 回転(theta)はYOLOv5では使わない。
        x, y, w, h, theta = ann['bbox']
        
        # 以下で物体の位置とサイズは画像のサイズに対する比で表す。

        # 物体の中心のXY座標
        x_center = (x + 0.5 * w) / image_width
        y_center = (y + 0.5 * h) / image_height

        # 物体のサイズ
        width    = w / image_width
        height   = h / image_height

        # ラベルファイル名
        label_file_name = img_inf["file_name"].replace('.jpg', '.txt')

        # ラベルファイルのパス
        label_file_path = f'{labels_dir}/{label_file_name}'

        # ラベルファイルをオープンする。
        with open(label_file_path, 'w') as f:

            # カテゴリー(クラス)のid, 物体の中心のXY座標, 物体のサイズを書く。
            f.write(f'{ann["category_id"]} {x_center} {y_center} {width} {height}\n')