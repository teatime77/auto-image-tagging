import os
import sys
import glob
import cv2
import PySimpleGUI as sg
from util import show_image, setPlaying
from odtk import ODTK
from yolo_v5 import YOLOv5
from main import parse, hue_shift, saturation_shift, value_shift, V_lo
from main import make_train_data, make_image_classes, make_training_data, get_video_capture

iterator = None
network = None
cap = None
playing = False

class_idx = 0
video_Idx = 0

def spin(label, key, val, min_val, max_val):
    return [ 
        sg.Text(label, size=(6,1)), sg.Text("", size=(6,1)), 
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(10, 1), key=key, enable_events=True )
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

def showImage(frame, gray_img, bin_img, clip_img, dst_img2, compo_img):

    # 原画を表示する。
    show_image(window['-image11-'], frame)

    # グレー画像を表示する。
    show_image(window['-image12-'], gray_img)

    # 二値画像を表示する。
    show_image(window['-image13-'], bin_img)

    # マスクした元画像を表示する。
    show_image(window['-image21-'], clip_img)

    show_image(window['-image22-'], dst_img2)

    show_image(window['-image23-'], compo_img)


def show_videos(class_idx, video_Idx):
    global cap

    # 背景画像ファイルのインデックス
    bg_img_idx = 0

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

                # 背景画像ファイルを読む。
                bg_img = cv2.imread(bg_img_paths[bg_img_idx])
                bg_img_idx = (bg_img_idx + 1) % len(bg_img_paths)

                frame, gray_img, bin_img, clip_img, dst_img2, compo_img, corners2, bounding_box = make_train_data(frame, bg_img)

                showImage(frame, gray_img, bin_img, clip_img, dst_img2, compo_img)

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

if __name__ == '__main__':
    video_dir, bg_img_dir, output_dir, network_name, data_size = parse()

    print(cv2.getBuildInformation())

    # 出力先フォルダを作る。
    os.makedirs(output_dir, exist_ok=True)

    # 背景画像ファイルのパス
    bg_img_paths = [ x for x in glob.glob(f'{bg_img_dir}/*') if os.path.splitext(x)[1] in [ '.jpg', '.png' ] ]

    image_classes = make_image_classes(video_dir)
    
    # ツリー表示のデータを作る。
    treedata = get_tree_data()

    sg.theme('DarkAmber')   # Add a touch of color

    # All the stuff inside your window.
    layout = [  
        [ sg.Tree(data=treedata,
                headings=[],
                auto_size_columns=True,
                num_rows=24,
                col0_width=30,
                key="-tree-",
                show_expanded=False,
                enable_events=True),
            sg.Column([
                [ sg.Image(filename='', size=(256,256), key='-image11-') ],
                [ sg.Image(filename='', size=(256,256), key='-image21-') ]
            ])
            ,
            sg.Column([
                [ sg.Image(filename='', size=(256,256), key='-image12-') ],
                [ sg.Image(filename='', size=(256,256), key='-image22-') ]
            ])
            ,
            sg.Column([
                [ sg.Image(filename='', size=(256,256), key='-image13-') ],
                [ sg.Image(filename='', size=(256,256), key='-image23-') ]
            ])
        ]
        ,
        [ sg.Slider(range=(0,100), default_value=0, size=(100,15), orientation='horizontal', change_submits=True, key='-img-pos-') ]
        ,
        [ sg.Frame('Color Augmentation', [
            spin('Hue', '-hue-shift-', hue_shift, 0, 30),
            spin('Saturation', '-saturation-shift-', saturation_shift, 0, 50),
            spin('Value', '-value-shift-', value_shift, 0, 50)
        ])]
        ,
        spin('V lo', '-Vlo-', V_lo, 0, 255)
        ,
        [ sg.Frame('Training Data', [
            [
                sg.Text('data size per class', size=(6,1)), 
                sg.Input(str(data_size), key='-data-size-', size=(6,1)),
                sg.Text('network', size=(6,1)), 
                sg.Combo(['ODTK', 'YOLOv5'], default_value = 'YOLOv5', key='-network-'),
                sg.Button('start', key='-save-all-')
            ]
        ])]
        ,
        [ sg.Button('Play', key='-play/pause-'), sg.Button('Close')] 
    ]

    # Create the Window
    window = sg.Window('Window Title', layout)

    cap = None
    iterator = None
    while True:
        # event, values = window.read()
        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
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

        elif event == '-Vlo-':
            V_lo = int(values[event])

        elif event == '-hue-shift-':
            hue_shift = int(values[event])

        elif event == '-saturation-shift-':
            saturation_shift = int(values[event])

        elif event == '-value-shift-':
            value_shift = int(values[event])

        elif event == '__TIMEOUT__':
            if playing and iterator is not None:
                try:
                    iterator.__next__()

                except StopIteration:
                    iterator = None

        elif event == '-img-pos-':
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(values['-img-pos-']))

                if not playing and iterator is not None:
                    try:
                        iterator.__next__()

                    except StopIteration:
                        iterator = None

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

            iterator = make_training_data(image_classes, bg_img_paths, network, data_size)

            total_data_size = data_size * len(image_classes)
            for idx, ret in enumerate(iterator):
                if not sg.one_line_progress_meter('make training network', idx+1, total_data_size, orientation='h'):

                    break

                if idx == total_data_size - 1:
                    sg.popup_ok('training data is created.')
            


        else:

            print('You entered ', event)

    window.close()
