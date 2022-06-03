import os
import sys
import datetime
import argparse
import cv2
import numpy as np
import PySimpleGUI as sg
from util import show_image, getContour


playing = False
writer  = None

# 明度の閾値(最小値)
v_min = 130


standard_resolutions = [
    [ 1280,  720 ],
    [  960,  720 ], 
    [ 1280,  960 ],
    [  800,  600 ], 
    [  800,  480 ], 
    [  640,  480 ]
]

def init_camera(camera_idx):
    global cap, brightness, frame_width, frame_height, frame_rate
    cap = cv2.VideoCapture(camera_idx)

    brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
    window['-brightness-'].update(value=brightness)

    print('BRIGHTNESS', brightness)

    for width, height in standard_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok = (cap.get(cv2.CAP_PROP_FRAME_WIDTH ) == width and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == height)
        print(width, height, ("OK" if ok else "NG"))
        if ok:
            break

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('WIDTH'     , frame_width)
    print('HEIGHT'    , frame_height)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)

def initWriter():
    """動画ファイルへのライターを初期化する。
    """
    global writer, write_path, write_cnt

    now = datetime.datetime.now()

    write_path = f'./capture/{now.strftime("%Y-%m-%d-%H-%M-%S")}.mp4'

    writer = cv2.VideoWriter(write_path, fmt, frame_rate, (frame_width, frame_height))

    write_cnt = 0

def readCap():
    global frame, write_cnt

    ret, frame = cap.read()

    if frame is None:
        return

    img = frame

    # 原画を表示する。
    show_image(window['-image1-'], img)

    # グレー画像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # 二値画像を表示する。
    bin_img = 255 - cv2.inRange(gray_img, v_min, 255)
    show_image(window['-image2-'], bin_img)

    # 二値化画像から輪郭とマスク画像を得る。
    msg, contour, mask_img = getContour(bin_img)

    window['-msg-'].update(value=msg)

    if msg != '':

        black_img = np.zeros(img.shape, dtype=np.uint8)
        show_image(window['-image3-'], black_img)
        return

    mask_img = mask_img[:, :, np.newaxis]
    mask_img = np.broadcast_to(mask_img, img.shape)

    white_img = np.full(img.shape, 255, dtype=np.uint8)
    clip_img = np.where(mask_img == 0, white_img, img)

    if writer is not None:
        # 画像を1フレーム分として書き込み
        writer.write(clip_img)

        write_cnt += 1

    # cv2.drawContours(clip_img, contour_family, -1, (255,0,0), 10)
    show_image(window['-image3-'], clip_img)

if __name__ == '__main__':

    # 動画/静止画を保存するフォルダを作る。
    os.makedirs('./capture', exist_ok=True)

    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f'camera {idx}')
            cap.release()
        else:
            num_cameras = idx
            break

    if num_cameras == 0:
        print('camera is not connexted')
        sys.exit(0)

    camera_list = [ f'カメラ {i+1}' for i in range(num_cameras) ]

    cap = None
    camera_idx = 0

    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)

    layout = [
        [
            sg.Column([
                [ 
                    sg.Frame('原画',[
                        [
                            sg.Image(filename='', size=(256,256), key='-image1-')
                        ]
                    ])
                ]
                ,
                [ 
                    sg.Frame('物体',[
                        [
                            sg.Image(filename='', size=(256,256), key='-image3-')
                        ]
                    ])
                ]
            ])
            ,
            sg.Column([
                [ 
                    sg.Frame('二値画像',[
                        [
                            sg.Image(filename='', size=(256,256), key='-image2-')
                        ]
                    ])
                ]
                ,
                [ 
                    sg.Text('カメラ', size=(12,1), pad=((10,0),(20,20)) ), 
                    sg.Combo(camera_list, default_value=camera_list[camera_idx], size=(8,1), enable_events=True, readonly=True, key='-camera-')
                ]
                ,
                [ 
                    sg.Text('明度の閾値', size=(12,1), pad=((10,0),(20,20)) ), 
                    sg.Spin(list(range(0, 255 + 1)), initial_value=v_min, size=(8, 1), key='-V-min-', enable_events=True )
                ]
                ,
                [ 
                    sg.Text('画像の明るさ', size=(12,1), pad=((10,0),(20,20)) ), 
                    sg.Spin(list(range(-1, 255 + 1)), initial_value=0, size=(8, 1), key='-brightness-', enable_events=True )
                ]
                ,
                [ 
                    sg.Button('動画撮影', key='-play/pause-', size=(8,1), pad=((10,0),(20,20)) ),
                    sg.Push(),
                    sg.Button('写真撮影', key='-shoot-')
                ]
                ,
                [sg.VPush()]
                ,
                [
                    sg.Push(),
                    sg.Button('閉じる', key='-close-')
                ]
            ], vertical_alignment='top', expand_y=True)
        ]
        ,
        [
            sg.Text('', key='-msg-', expand_x=True, relief=sg.RELIEF_SUNKEN ), 
        ]
    ]

    window = sg.Window('カメラ', layout, font='Any 20')

    while True:

        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED or event == '-close-':
            cap.release()
            break

        elif event == '-camera-':
            cap.release()

            camera_idx = camera_list.index(values[event])
            init_camera(camera_idx)

        elif event == '-play/pause-':
            playing = not playing

            if playing:

                window['-play/pause-'].update(text='停止')
                initWriter()

            else:

                window['-play/pause-'].update(text='動画撮影')
                writer.release()
                writer = None

                if write_cnt == 0:

                    sg.popup('物体の画像がありません。')

                    # 動画ファイルを削除する。
                    os.remove(write_path)

        elif event == '-V-min-':
            # 明度の閾値(最小値)
            v_min = int(values[event])

        elif event == '-brightness-':
            brightness = int(values[event])
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

        elif event == '-shoot-':
            now = datetime.datetime.now()
            img_path = f'./capture/{now.strftime("%Y-%m-%d-%H-%M-%S")}.jpg'
            cv2.imwrite(img_path, frame)

        elif event == '__TIMEOUT__':
            if cap is None:
                camera_idx = 0
                init_camera(camera_idx)

            readCap()
