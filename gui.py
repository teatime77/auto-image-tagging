import os
import sys
import glob
import cv2
import PySimpleGUI as sg
from util import spin, show_image, getContour, edge_width, setPlaying
from odtk import _corners2rotatedbbox, ODTK
from yolo_v5 import YOLOv5
from main import parse, data_size, hue_shift, saturation_shift, value_shift, V_lo
from main import imageClasses, make_train_data, make_image_classes

network = None

# 背景画像ファイルのインデックス
bgImgIdx = 0

def initCap():
    """動画のキャプチャーの初期処理をする。

    Returns:
        VideoCapture: キャプチャー オブジェクト
    """
    global VideoIdx, playing

    # 動画ファイルのパス
    video_path = imageClasses[classIdx].videoPathes[VideoIdx]

    # 動画のキャプチャー オブジェクト
    cap = cv2.VideoCapture(video_path)    

    if not cap.isOpened():
        print("動画再生エラー")
        sys.exit()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window['-img-pos-'].update(range=(0, frame_count))
    window['-img-pos-'].update(value=0)
    print(f'再生開始 フレーム数:{cap.get(cv2.CAP_PROP_FRAME_COUNT)} {os.path.basename(video_path)}')

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

def readCap():
    global cap, VideoIdx, classIdx, bgImgIdx

    ret, frame = cap.read()
    if ret:
        # 画像が取得できた場合

        # 動画の現在位置
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 動画の現在位置の表示を更新する。
        window['-img-pos-'].update(value=pos)

        # 背景画像ファイルを読む。
        bg_img = cv2.imread(bgImgPaths[bgImgIdx])
        bgImgIdx = (bgImgIdx + 1) % len(bgImgPaths)

        frame, gray_img, bin_img, clip_img, dst_img2, compo_img, corners2, bounding_box = make_train_data(frame, bg_img)

        showImage(frame, gray_img, bin_img, clip_img, dst_img2, compo_img)

        if network is not None:
            # 保存中の場合

            # 動画の現在位置
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

            network.add_image(classIdx, VideoIdx, pos, compo_img, corners2, bounding_box)

            # 取得画像枚数
            images_cnt = network.images_cnt()

            window['-images-cnt-'].update(f'  {images_cnt}枚')

            # 現在のクラスのテータ数をカウントアップ
            class_data_cnt[classIdx] += 1

            if data_size <= class_data_cnt[classIdx]:
                # 現在のクラスのテータ数が指定値に達した場合

                # キャプチャー オブジェクトを解放する。
                cap.release()

                if data_size <= min(class_data_cnt):
                    # すべてのクラスのデータ数が指定値に達した場合

                    stopSave()
                    print("保存終了")

                else:
                    # データ数が指定値に達していないクラスがある場合

                    # データ数が最小のクラスのインデックス
                    classIdx = class_data_cnt.index(min(class_data_cnt))

                    # 動画のインデックス
                    VideoIdx = 0

                    cap = initCap()

    else:
        # 動画の終わりの場合

        # 動画のインデックスをカウントアップ
        VideoIdx += 1

        # キャプチャー オブジェクトを解放する。
        cap.release()

        if VideoIdx < len(imageClasses[classIdx].videoPathes):
            # 同じクラスの別の動画ファイルがある場合

            cap = initCap()
        else:
            # 同じクラスの別の動画ファイルがない場合

            # データ数が最小のクラスのインデックス
            classIdx = class_data_cnt.index(min(class_data_cnt))

            # 動画のインデックス
            VideoIdx = 0

            cap = initCap()


def stopSave():
    global VideoIdx, classIdx, network, playing

    network.save()

    VideoIdx = 0
    classIdx = 0
    network = None

    playing = setPlaying(window, False)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def saveImgs():
    global cap, VideoIdx, classIdx

    VideoIdx = 0

    cap = initCap()

def showImgPos():
    if cap is not None:
        pos = int(values['-img-pos-'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        readCap()

def get_tree_data():
    treedata = sg.TreeData()

    # すべてのクラスに対し
    for img_class in imageClasses:
        treedata.Insert('', img_class.name, img_class.name, values=[])

        # クラスの動画に対し
        for video_path in img_class.videoPathes:
            video_name = os.path.basename(video_path)
            treedata.Insert(img_class.name, video_path, video_name, values=[video_path])

    return treedata

if __name__ == '__main__':
    video_dir, bg_img_dir, output_dir, network_name = parse()

    print(cv2.getBuildInformation())

    # 出力先フォルダを作る。
    os.makedirs(output_dir, exist_ok=True)

    # 背景画像ファイルのパス
    bgImgPaths = [ x for x in glob.glob(f'{bg_img_dir}/*') if os.path.splitext(x)[1] in [ '.jpg', '.png' ] ]

    imageClasses = make_image_classes(video_dir)


    # ツリー表示のデータを作る。
    treedata = get_tree_data()

    sg.theme('DarkAmber')   # Add a touch of color

    # All the stuff inside your window.
    layout = [  
        [ sg.Tree(data=treedata,
                headings=[],
                auto_size_columns=True,
                num_rows=24,
                col0_width=50,
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
        [ sg.Input(str(data_size), key='-data-size-', size=(6,1)), sg.Text('', size=(6,1), key='-images-cnt-') ]
        ,
        [ sg.Frame('Color Augmentation', [
            spin('Hue', '-hue-shift-', hue_shift, 0, 30),
            spin('Saturation', '-saturation-shift-', saturation_shift, 0, 50),
            spin('Value', '-value-shift-', value_shift, 0, 50)
        ])]
        ,
        spin('V lo', '-Vlo-', V_lo, 0, 255),
        [ sg.Text('network', size=(6,1)), sg.Combo(['ODTK', 'YOLOv5'], default_value = 'YOLOv5', key='-network-') ],
        [ sg.Button('Play', key='-play/pause-'), sg.Button('Save All', key='-save-all-'), sg.Button('Close')] ]

    # Create the Window
    window = sg.Window('Window Title', layout)

    cap = None
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

                if cap is not None:
                    # 再生中の場合

                    # キャプチャー オブジェクトを解放する。
                    cap.release()


                # 動画ファイルを含むクラスとインデックス
                classIdx, img_class = [ (idx, c) for idx, c in enumerate(imageClasses) if video_path in c.videoPathes ][0]

                VideoIdx  = img_class.videoPathes.index(video_path)

                cap = initCap()

        elif event == '-Vlo-':
            V_lo = int(values[event])

        elif event == '-hue-shift-':
            hue_shift = int(values[event])

        elif event == '-saturation-shift-':
            saturation_shift = int(values[event])

        elif event == '-value-shift-':
            value_shift = int(values[event])

        elif event == '__TIMEOUT__':
            if cap is not None and playing:
                readCap()

        elif event == '-img-pos-':
            showImgPos()

        elif event == '-play/pause-':
            playing = setPlaying(window, not playing)

        elif event == '-save-all-':
            data_size = int(values['-data-size-'])
            class_data_cnt = [0] * len(imageClasses)
            classIdx = 0

            # 背景画像ファイルのインデックス
            bgImgIdx = 0

            if values['-network-'] == 'ODTK':
                network = ODTK(output_dir, imageClasses)
            else:
                network = YOLOv5(output_dir, imageClasses)

            saveImgs()

        else:

            print('You entered ', event)

    window.close()
