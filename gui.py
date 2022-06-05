import os
import sys
import glob
import numpy as np
import cv2
import PySimpleGUI as sg
from util import show_image
from odtk import ODTK
from main import parse
from main import make_train_data, make_image_classes, make_training_data, get_video_capture

iterator = None
cap = None
playing = False
show_rect = True
use_same_bg_img = False

v_min = 130
"""明度の閾値(最小値)"""

hue_shift = 10
saturation_shift = 15
value_shift = 15


class_idx = 0
video_Idx = 0

def spin(label : str, key : str, val : int, min_val : int, max_val : int):
    """TextとSpinのペアを作る。

    Args:
        label : Textのラベル
        key : Spinのキー
        val : Spinの初期値
        min_val : 最小値
        max_val : 最大値

    Returns: TextとSpinのペア
    """
    return [ 
        sg.Text(label, size=(12,1)),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(5, 1), key=key, enable_events=True )
    ]

def setPlaying(window, is_playing):
    if is_playing:
        window['-play/pause-'].update(text='休止')
    else:
        window['-play/pause-'].update(text='再生')

    return is_playing

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

    # 動画ファイルのフレーム数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 再生位置のスライダーの値の範囲を更新する。
    window['-img-pos-'].update(range=(0, frame_count - 1))

    # 再生位置のスライダーの現在位置を更新する。
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

                # 色相、彩度、明度の変化量
                hsv_shift = (hue_shift, saturation_shift, value_shift)

                bin_img, mask_img, compo_img, aug_img, box, corners2, bounding_box = make_train_data(frame, bg_img, img_size, v_min, hsv_shift)
                if mask_img is None:

                    black_img = np.zeros(frame.shape, dtype=np.uint8)
                    for img, key in zip( [frame, bin_img, black_img, black_img], [ '-image11-', '-image12-', '-image21-', '-image22-']):
                        show_image(window[key], img)
                    yield
                    continue

                # 原画をマスクでクリップする。
                mask3_img = np.broadcast_to(mask_img[:, :, np.newaxis], aug_img.shape)
                clip_img = np.where(mask3_img == 0, mask3_img, aug_img)

                compo_img  = compo_img.copy()

                if show_rect:
                    # 矩形を表示する場合

                    # 外接矩形を描く。
                    cv2.drawContours(clip_img, [ np.int0(box) ], 0, (255,0,0), 3)

                    x, y, w, h, theta = bounding_box

                    x, y, w, h = np.int32((x, y, w, h))


                    # 座標変換後の外接矩形を描く。
                    cv2.drawContours(compo_img, [ np.int0(corners2)  ], 0, (255,0,0), 3)

                    # バウンディングボックスを描く。
                    cv2.rectangle(compo_img, (int(x),int(y)), (int(x+w),int(y+h)), (0,0,255), 3)

                    # バウンディングボックスの左上の頂点の位置に円を描く。
                    cv2.circle(compo_img, (int(x), int(y)), 5, (255,255,255), -1)


                # 原画を表示する。
                show_image(window['-image11-'], frame)

                # 二値画像を表示する。
                show_image(window['-image12-'], bin_img)

                # 原画をマスクでクリップした画像を表示する。
                show_image(window['-image21-'], clip_img)

                # 合成画像を表示する。
                show_image(window['-image22-'], compo_img)

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
    """TreeDataを作り、クラスと動画ファイルのノードを挿入する。

    Returns: TreeData
    """
    treedata = sg.TreeData()

    # すべてのクラスに対し
    for img_class in image_classes:

        # TreeDataにクラスのノードを挿入する。
        treedata.Insert('', img_class.name, img_class.name, values=[])

        # クラスの動画に対し
        for video_path in img_class.videoPathes:
            video_name = os.path.basename(video_path)

            # TreeDataに動画ファイルのノードを挿入する。
            treedata.Insert(img_class.name, video_path, video_name, values=[video_path])

    return treedata

def update_one_frame(event : str):
    """現在のフレームの表示を更新する。

    Args:
        event : イベント
    """
    global iterator, use_same_bg_img

    if cap is not None:
        # 再生中の場合

        # 再生位置 = Sliderの値 - 1
        pos = max(0, int(values['-img-pos-']) - 1)

        # 再生位置をセットする。
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        # Sliderのイベントでなければ、同じ背景画像を使う。
        use_same_bg_img = (event != '-img-pos-')

        if not playing and iterator is not None:
            try:
                iterator.__next__()

            except StopIteration:
                iterator = None

        use_same_bg_img = False


if __name__ == '__main__':
    video_dir, bg_img_dir, output_dir, data_size, img_size, v_min, hsv_shift = parse()

    # 色相、彩度、明度の変化量
    hue_shift, saturation_shift, value_shift = hsv_shift

    print(cv2.getBuildInformation())

    # 出力先フォルダを作る。
    os.makedirs(output_dir, exist_ok=True)

    # 背景画像ファイルのパス
    bg_img_paths = [ x for x in glob.glob(f'{bg_img_dir}/*') if os.path.splitext(x)[1] in [ '.jpg', '.png' ] ]

    # 画像のクラスのリスト
    image_classes = make_image_classes(video_dir)
    
    # ツリー表示のデータを作る。
    treedata = get_tree_data()

    dsp_size = 360
    font_str = 'Any 12'

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
                        spin('明度の閾値', '-V-min-', v_min, 0, 255)
                    ],  expand_x=True, pad=((0,0),(10,10)) )
                ]
                ,
                [  
                    sg.Frame('データ拡張', [
                        spin('色相', '-hue-shift-', hue_shift, 0, 30),
                        spin('彩度', '-saturation-shift-', saturation_shift, 0, 50),
                        spin('明度', '-value-shift-', value_shift, 0, 50)
                    ],  expand_x=True, pad=((0,0),(10,10)) )
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
                            sg.Push(),
                            sg.Button('作成開始', key='-save-all-', pad=((50,0),(0,0)))
                        ]
                    ],  expand_x=True, pad=((0,0),(10,10)) )
                ]
                ,
                [  
                    sg.Frame('表示', [
                        [ sg.Checkbox('矩形を表示', default=show_rect, enable_events=True, key='-show-rect-') ]
                    ],  expand_x=True)
                ]
            ], vertical_alignment='top', expand_y=True)
            ,
            sg.Column([
                [
                    sg.Column([

                        [ 
                            sg.Frame('原画',[
                                [ sg.Image(filename='', size=(dsp_size,dsp_size), key='-image11-') ]
                            ])
                        ]
                        ,
                        [ 
                            sg.Frame('物体',[
                                [ sg.Image(filename='', size=(dsp_size,dsp_size), key='-image21-') ]
                            ])
                        ]
                    ])
                    ,
                    sg.Column([

                        [ 
                            sg.Frame('二値画像',[
                                [ sg.Image(filename='', size=(dsp_size,dsp_size), key='-image12-') ]
                            ])
                        ]
                        ,
                        [ 
                            sg.Frame('合成画像',[
                                [ sg.Image(filename='', size=(dsp_size,dsp_size), key='-image22-') ]
                            ])
                        ]
                    ])
                ]
            ])
        ]
        ,
        [ 
            sg.Button('再生', key='-play/pause-', size=(6,1)), 
            sg.Slider(range=(0,100), default_value=0, orientation='horizontal', change_submits=True, key='-img-pos-', expand_x=True),
        ]
        ,
        [
            sg.Text('', key='-msg-', expand_x=True, relief=sg.RELIEF_SUNKEN ), 
        ]
        ,
        [
            sg.Push(),
            sg.Button('閉じる', key='-close-', size=(6,1), pad=((0,0),(10,0)))
        ]
    ]

    # メインウィンドウ
    window = sg.Window('auto img tab', layout, font=font_str)

    cap = None
    iterator = None
    while True:
        # event, values = window.read()
        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED or event=='-close-':
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
            # 明度の閾値(最小値)のスピン

            # 明度の閾値
            v_min = int(values[event])

            # 現在のフレームの表示を更新する。
            update_one_frame(event)

        elif event == '-hue-shift-':
            # 色相の変化量のスピン

            # 色相の変化量
            hue_shift = int(values[event])

            # 現在のフレームの表示を更新する。
            update_one_frame(event)

        elif event == '-saturation-shift-':
            # 彩度の変化量のスピン

            # 彩度の変化量
            saturation_shift = int(values[event])

            # 現在のフレームの表示を更新する。
            update_one_frame(event)

        elif event == '-value-shift-':
            # 明度の変化量のスピン

            # 明度の変化量
            value_shift = int(values[event])

            # 現在のフレームの表示を更新する。
            update_one_frame(event)

        elif event == '-show-rect-':
            # 矩形の表示/非表示のチェックボックス

            # 矩形の表示/非表示
            show_rect = values[event]

            # 現在のフレームの表示を更新する。
            update_one_frame(event)

        elif event == '__TIMEOUT__':
            if playing and iterator is not None:
                try:
                    iterator.__next__()

                except StopIteration:
                    iterator = None

        elif event == '-img-pos-':

            # 現在のフレームの表示を更新する。
            update_one_frame(event)

        elif event == '-play/pause-':
            # 再生/休止ボタン

            playing = setPlaying(window, not playing)

            if playing and iterator is None:
                iterator = show_videos(class_idx, video_Idx)

        elif event == '-save-all-':
            data_size = int(values['-data-size-'])

            # 色相、彩度、明度の変化量
            hsv_shift = (hue_shift, saturation_shift, value_shift)

            # 全クラスの学習データ数
            total_data_size = data_size * len(image_classes)

            for idx, ret in enumerate( make_training_data(output_dir, image_classes, bg_img_paths, data_size, img_size, v_min, hsv_shift) ):
                if not sg.one_line_progress_meter('学習データ作成', idx+1, total_data_size, orientation='h'):
                    # Cancelボタンがクリックされた場合

                    break

                if idx == total_data_size - 1:
                    # 最後のデータの場合
                    
                    sg.popup_ok('学習データが作成されました。')
                    break
            
        else:

            print('You entered ', event)

    window.close()
