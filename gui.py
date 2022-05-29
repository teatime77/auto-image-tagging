import os
import sys
import glob
import numpy as np
import cv2
import PySimpleGUI as sg
from util import show_image, setPlaying
from odtk import ODTK
from yolo_v5 import YOLOv5
from main import parse
from main import make_train_data, make_image_classes, make_training_data, get_video_capture

iterator = None
network = None
cap = None
playing = False
show_rect = True
use_same_bg_img = False

# 明度の閾値(最小値)
v_min = 130

hue_shift = 10
saturation_shift = 15
value_shift = 15


class_idx = 0
video_Idx = 0

def spin(label, key, val, min_val, max_val):
    return [ 
        sg.Text(label, size=(8,1)),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(5, 1), key=key, enable_events=True )
    ]

def init_capture(class_idx, video_Idx):
    """動画のキャプチャーの初期処理をする。

    Returns:
        VideoCapture: キャプチャー オブジェクト
    """
    global playing

    # 動画ファイルのパス
    video_path = image_classes[class_idx].videoPathes[video_Idx]

    # 動画のキャプチャー オブジェクト
    cap = get_video_capture(video_path)    

    if not cap.isOpened():
        print("動画再生エラー")
        sys.exit()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window['-img-pos-'].update(range=(0, frame_count - 1))
    window['-img-pos-'].update(value=0)

    playing = setPlaying(window, True)

    return cap


def show_videos(class_idx, video_Idx):
    global cap

    # 背景画像ファイルのインデックス
    bg_img_idx = 0

    prev_bg_img = None

    while class_idx < len(image_classes):
        cap = init_capture(class_idx, video_Idx)
        print('init cap')

        while True:
            ret, frame = cap.read()
            if ret:
                # 画像が取得できた場合

                # 動画の現在位置
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 動画の現在位置の表示を更新する。
                window['-img-pos-'].update(value=pos)

                if use_same_bg_img and prev_bg_img is not None:
                    bg_img = prev_bg_img

                else:
                    # 背景画像ファイルを読む。
                    bg_img = cv2.imread(bg_img_paths[bg_img_idx])
                    bg_img_idx = (bg_img_idx + 1) % len(bg_img_paths)

                prev_bg_img = bg_img

                hsv_shift = (hue_shift, saturation_shift, value_shift)
                bin_img, mask_img, compo_img, aug_img, box, corners2, bounding_box = make_train_data(frame, bg_img, img_size, v_min, hsv_shift)
                if mask_img is None:

                    black_img = np.zeros(frame.shape, dtype=np.uint8)
                    for img, key in zip( [frame, bin_img, black_img, black_img], [ '-image11-', '-image12-', '-image21-', '-image22-']):
                        show_image(window[key], img)
                    yield
                    continue

                # 元画像にマスクをかける。
                mask3_img = np.broadcast_to(mask_img[:, :, np.newaxis], aug_img.shape)
                clip_img = np.where(mask3_img == 0, mask3_img, aug_img)

                compo_img  = compo_img.copy()

                if show_rect:

                    # 外接矩形を描く。
                    cv2.drawContours(clip_img, [ np.int0(box) ], 0, (0,255,0), 2)

                    x, y, w, h, theta = bounding_box

                    x, y, w, h = np.int32((x, y, w, h))


                    # 座標変換後の外接矩形を描く。
                    cv2.drawContours(compo_img, [ np.int0(corners2)  ], 0, (0,255,0), 2)

                    # バウンディングボックスを描く。
                    cv2.rectangle(compo_img, (int(x),int(y)), (int(x+w),int(y+h)), (0,0,255), 3)

                    # バウンディングボックスの左上の頂点の位置に円を描く。
                    cv2.circle(compo_img, (int(x), int(y)), 10, (255,255,255), -1)

                for img, key in zip( [frame, bin_img, clip_img, compo_img],
                                     [ '-image11-', '-image12-', '-image21-', '-image22-' ]):
                    show_image(window[key], img)

                yield
                continue

            else:
                # 動画の終わりの場合

                break

        # 動画のインデックス
        video_Idx += 1

        if len(image_classes[class_idx].videoPathes) <= video_Idx:
            # 同じクラスの別の動画ファイルがない場合

            # クラスのインデックス
            class_idx += 1

            # 動画のインデックス
            video_Idx = 0


def get_tree_data():
    treedata = sg.TreeData()

    # すべてのクラスに対し
    for img_class in image_classes:
        treedata.Insert('', img_class.name, img_class.name, values=[])

        # クラスの動画に対し
        for video_path in img_class.videoPathes:
            video_name = os.path.basename(video_path)
            treedata.Insert(img_class.name, video_path, video_name, values=[video_path])

    return treedata

def update_one_frame(event : str):
    """現在のフレームの表示を更新する。

    Args:
        event : イベント
    """
    global iterator, use_same_bg_img

    if cap is not None:
        pos = max(0, int(values['-img-pos-']) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        use_same_bg_img = (event != '-img-pos-')

        if not playing and iterator is not None:
            try:
                iterator.__next__()

            except StopIteration:
                iterator = None

        use_same_bg_img = False


if __name__ == '__main__':
    video_dir, bg_img_dir, output_dir, network_name, data_size, img_size, v_min, hsv_shift = parse()
    hue_shift, saturation_shift, value_shift = hsv_shift

    print(cv2.getBuildInformation())

    # 出力先フォルダを作る。
    os.makedirs(output_dir, exist_ok=True)

    # 背景画像ファイルのパス
    bg_img_paths = [ x for x in glob.glob(f'{bg_img_dir}/*') if os.path.splitext(x)[1] in [ '.jpg', '.png' ] ]

    image_classes = make_image_classes(video_dir)
    
    # ツリー表示のデータを作る。
    treedata = get_tree_data()

    sg.theme('DarkAmber')   # Add a touch of color

    dsp_size = 360
    # All the stuff inside your window.
    layout = [  
        [ 
            sg.Column([
                [  
                    sg.Tree(data=treedata,
                        headings=[],
                        auto_size_columns=True,
                        num_rows=24,
                        col0_width=30,
                        key="-tree-",
                        show_expanded=False,
                        enable_events=True)
                ]
                ,
                [  
                    sg.Frame('二値化', [
                        spin('明度 閾値', '-V-min-', v_min, 0, 255)
                    ],  expand_x=True)
                ]
                ,
                [  
                    sg.Frame('データ拡張', [
                        spin('色相', '-hue-shift-', hue_shift, 0, 30),
                        spin('彩度', '-saturation-shift-', saturation_shift, 0, 50),
                        spin('明度', '-value-shift-', value_shift, 0, 50)
                    ],  expand_x=True)
                ]
                ,
                [  
                    sg.Frame('学習データ', [
                        [
                            sg.Text('1クラス当たりのデータ数'), 
                            sg.Input(str(data_size), key='-data-size-', size=(6,1))
                        ]
                        ,
                        [
                            sg.Text('深層学習', size=(8,1)), 
                            sg.Combo(['ODTK', 'YOLOv5'], default_value = 'YOLOv5', key='-network-', size=(8,1)),
                            sg.Button('作成開始', key='-save-all-', pad=((50,0),(0,0)))
                        ]
                    ],  expand_x=True)
                ]
                ,
                [  
                    sg.Frame('表示', [
                        [ sg.Checkbox('矩形を表示', default=show_rect, enable_events=True, key='-show-rect-') ]
                    ],  expand_x=True)
                ]
            ])
            ,
            sg.Column([
                [
                    sg.Column([
                        [ sg.Image(filename='', size=(dsp_size,dsp_size), key='-image11-') ],
                        [ sg.Image(filename='', size=(dsp_size,dsp_size), key='-image21-') ]
                    ])
                    ,
                    sg.Column([
                        [ sg.Image(filename='', size=(dsp_size,dsp_size), key='-image12-') ],
                        [ sg.Image(filename='', size=(dsp_size,dsp_size), key='-image22-') ]
                    ])
                ]
            ])
        ]
        ,
        [ 
            sg.Button('Play ', key='-play/pause-'), 
            sg.Slider(range=(0,100), default_value=0, size=(111,15), orientation='horizontal', change_submits=True, key='-img-pos-')
        ]
    ]

    # Create the Window
    window = sg.Window('Window Title', layout)

    cap = None
    iterator = None
    while True:
        # event, values = window.read()
        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED:
            break

        if event == '-tree-':
            print(f'クリック [{values[event]}] [{values[event][0]}]')

            # クリックされたノードのvaluesの最初の値
            video_path = values[event][0]

            if os.path.isfile(video_path):
                # 動画ファイルの場合

                # 動画ファイルを含むクラスとインデックス
                class_idx, img_class = [ (idx, c) for idx, c in enumerate(image_classes) if video_path in c.videoPathes ][0]

                video_Idx  = img_class.videoPathes.index(video_path)

                iterator = show_videos(class_idx, video_Idx)
                playing = setPlaying(window, True)

        elif event == '-V-min-':
            # 明度の閾値(最小値)
            v_min = int(values[event])
            update_one_frame(event)

        elif event == '-hue-shift-':
            hue_shift = int(values[event])
            update_one_frame(event)

        elif event == '-saturation-shift-':
            saturation_shift = int(values[event])
            update_one_frame(event)

        elif event == '-value-shift-':
            value_shift = int(values[event])
            update_one_frame(event)

        elif event == '-show-rect-':
            show_rect = values[event]
            update_one_frame(event)

        elif event == '__TIMEOUT__':
            if playing and iterator is not None:
                try:
                    iterator.__next__()

                except StopIteration:
                    iterator = None

        elif event == '-img-pos-':
            update_one_frame(event)

        elif event == '-play/pause-':
            playing = setPlaying(window, not playing)

            if playing and iterator is None:
                iterator = show_videos(class_idx, video_Idx)

        elif event == '-save-all-':
            data_size = int(values['-data-size-'])

            if values['-network-'] == 'ODTK':
                network = ODTK(output_dir, image_classes)

            elif values['-network-'] == 'YOLOv5':
                network = YOLOv5(output_dir, image_classes)

            else:
                assert(False)

            hsv_shift = (hue_shift, saturation_shift, value_shift)
            iterator = make_training_data(image_classes, bg_img_paths, network, data_size, img_size, v_min, hsv_shift)

            total_data_size = data_size * len(image_classes)
            for idx, ret in enumerate(iterator):
                if not sg.one_line_progress_meter('make training network', idx+1, total_data_size, orientation='h'):

                    break

                if idx == total_data_size - 1:
                    sg.popup_ok('training data is created.')
                    break
            
            iterator = None



        else:

            print('You entered ', event)

    window.close()
