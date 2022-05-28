import os
import sys
import datetime
import argparse
import cv2
import numpy as np
import PySimpleGUI as sg
from util import show_image, getContour, setPlaying
from gui import spin
from main import make_train_data

playing = False
writer  = None
V_lo = 253

def initWriter():
    """動画ファイルへのライターを初期化する。
    """
    global writer

    os.makedirs(f'{output_dir}', exist_ok=True)

    now = datetime.datetime.now()

    file_path = f'{output_dir}/{now.strftime("%Y-%m-%d-%H-%M-%S")}.mp4'

    writer = cv2.VideoWriter(file_path, fmt, frame_rate, (frame_width, frame_height)) # ライター作成

def readCap():
    ret, frame = cap.read()

    if frame is None:
        return

    img = frame

    show_image(window['-image11-'], img)

    # グレー画像を表示する。
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    show_image(window['-image12-'], gray_img)

    # 二値画像を表示する。
    bin_img = 255 - cv2.inRange(gray_img, V_lo, 255)
    show_image(window['-image22-'], bin_img)

    contour, mask_img, edge_img = getContour(bin_img)
    if contour is None:
        return

    mask_img = mask_img[:, :, np.newaxis]
    mask_img = np.broadcast_to(mask_img, img.shape)

    white_img = np.full(img.shape, 255, dtype=np.uint8)
    clip_img = np.where(mask_img == 0, white_img, img)

    if writer is not None:
        # 画像を1フレーム分として書き込み
        writer.write(clip_img)

    # cv2.drawContours(clip_img, contour_family, -1, (255,0,0), 10)
    show_image(window['-image21-'], clip_img)


resolutions = [
    [  160,  120 ], 
    [  352,  288 ], 
    [  640,  360 ], 
    [  640,  480 ], 
    [  800,  600 ], 
    [ 1024,  576 ], 
    [  960,  720 ], 
    [ 1280,  720 ], 
    [ 1600,  896 ], 
    [ 1920, 1080 ]
]

def parse():
    parser = argparse.ArgumentParser(description='Auto Image Tag')
    parser.add_argument('-o','--output', type=str, help='path to outpu', default='capture')
    parser.add_argument('-imsz', '--img_size', type=int, help='image size', default=720)

    args = parser.parse_args(sys.argv[1:])

    # 出力先フォルダのパス
    output_dir = args.output.replace('\\', '/')

    return output_dir, args.img_size

if __name__ == '__main__':
    output_dir, img_size = parse()

    cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する

    brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
    exposure   = int(cap.get(cv2.CAP_PROP_EXPOSURE))
    contrast   = int(cap.get(cv2.CAP_PROP_CONTRAST))

    print('BRIGHTNESS', brightness)
    print('EXPOSURE'  , exposure)
    print('CONTRAST'  , contrast)
    print('FPS'       , cap.get(cv2.CAP_PROP_FPS))
    print('AUTO EXPOSURE', cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))

    # WIDTH  = 960
    # HEIGHT = 720

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH )) == WIDTH
    # assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == HEIGHT

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # img_size   = min(WIDTH, HEIGHT)

    print('WIDTH'     , frame_width)
    print('HEIGHT'    , frame_height)


    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)

    # ret_val , cap_for_exposure = cap.read()
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # cap.set(cv2.CAP_PROP_EXPOSURE , -1)


    sg.theme('DarkAmber')   # Add a touch of color

    layout = [
        [
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
        spin('V lo', '-Vlo-', V_lo, 0, 255),
        spin('brightness', '-brightness-', brightness, 0, 255),
        spin('exposure', '-exposure-', exposure, -20, 20),
        spin('contrast', '-contrast-', contrast,   0, 255)
        ,
        [ sg.Button('Play', key='-play/pause-'), sg.Button('Close') ]
    ]

    window = sg.Window('Window Title', layout)

    while True:

        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED or event == 'Close':
            break

        elif event == '-play/pause-':
            playing = setPlaying(window, not playing)

            if playing:
                initWriter()

            else:
                writer.release()
                writer = None

        elif event == '-Vlo-':
            V_lo = int(values[event])

        elif event == '-brightness-':
            brightness = int(values[event])
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

        elif event == '-exposure-':
            exposure = int(values[event])
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        elif event == '-contrast-':
            contrast = int(values[event])
            cap.set(cv2.CAP_PROP_CONTRAST, contrast)

        elif event == '__TIMEOUT__':
            readCap()
