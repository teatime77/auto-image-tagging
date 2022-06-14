import os
import sys
import datetime
import cv2
import numpy as np
import PySimpleGUI as sg
from util import show_image, getContour

cap = None
"""カメラのキャプチャ オブジェクト"""

writer  = None
"""動画ファイルへのライター"""

v_min = 130
"""明度の閾値(最小値)"""


standard_resolutions = [
    [ 1280,  720 ],
    [  960,  720 ], 
    [ 1280,  960 ],
    [  800,  600 ], 
    [  800,  480 ], 
    [  640,  480 ]
]
"""代表的な解像度のリスト"""

make_square = True
# 画像を正方形にする

chroma_key_hue = -1
"""背景の色相"""

hue_range = 5
# 色相の範囲

bin_method = ''
"""明度か色相の指定"""

def get_num_cameras():
    """パソコンに接続されたカメラの数を返す。

    Returns: カメラの数
    """
    for idx in range(10):

        # カメラのキャプチャ オブジェクト
        cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            # カメラに接続できた場合

            # カメラのキャプチャ オブジェクトを解放する。
            cap.release()
        else:
            # カメラに接続できなかった場合

            return idx

def init_camera(camera_idx):
    """カメラの初期処理をする。

    Args:
        camera_idx : カメラのインデックス (0オリジン)
    """
    global cap, brightness, frame_width, frame_height, img_size, frame_rate

    sg.popup_quick_message('カメラの初期処理をしています．．．', font='Any 20', background_color='blue')

    # カメラのキャプチャ オブジェクト
    cap = cv2.VideoCapture(camera_idx)

    # 画像の明るさ
    brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))

    # 画像の明るさの表示を更新する。
    window['-brightness-'].update(value=brightness)

    # 代表的な解像度に対して
    for width, height in standard_resolutions:
        
        # 画像の幅と高さを設定する。
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if cap.get(cv2.CAP_PROP_FRAME_WIDTH ) == width and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == height:
            # 設定した値が反映された場合

            break

    # 画像の幅と高さ
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    img_size = min(frame_width, frame_height)

    # フレームレート
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

def initWriter():
    """動画ファイルへのライターを初期化する。
    """
    global writer, write_path, write_cnt

    now = datetime.datetime.now()

    # 動画ファイルのパス
    write_path = f'./capture/{now.strftime("%Y-%m-%d-%H-%M-%S")}.mp4'

    # 動画ファイルの形式(ここではmp4)
    video_format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # 動画ファイルへのライター
    if make_square:
        writer = cv2.VideoWriter(write_path, video_format, frame_rate, (img_size, img_size))
    else:
        writer = cv2.VideoWriter(write_path, video_format, frame_rate, (frame_width, frame_height))

    # 動画ファイルへの書き込み件数
    write_cnt = 0

def read_one_frame():
    """カメラから1フレームを読む。
    """
    global frame, write_cnt

    # カメラから1フレームを読む。
    ok, frame = cap.read()
    if not ok:
        # 読み込めなかった場合

        return

    if make_square:
        x = (frame_width  - img_size) // 2
        y = (frame_height - img_size) // 2
        frame = frame[y:(y+img_size), x:(x+img_size), :]

    # 原画を表示する。
    show_image(window['-image1-'], frame)

    # 白画像
    white_img = np.full(frame.shape, 255, dtype=np.uint8)

    if bin_method == '明度':
        # 明度で二値画像を作る場合

        # グレー画像
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        # 二値画像
        bin_img = 255 - cv2.inRange(gray_img, v_min, 255)

    elif bin_method == '色相' and chroma_key_hue != -1:
        # 色相で二値画像を作り、背景の色相が指定済みの場合

        # HSV画像
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

        # 色相
        hue_img = hsv_img[:, :, 0]

        if 180 < chroma_key_hue + hue_range:
            # 色相の上限が180を超える場合

            # 下限から180までのマスク
            mask1 = cv2.inRange(hue_img, chroma_key_hue - hue_range, 180)

            # 0から上限までのマスク
            mask2 = cv2.inRange(hue_img, 0, chroma_key_hue + hue_range - 180)

            # 2つのマスクの和
            mask  = cv2.bitwise_or(mask1, mask2)

        elif chroma_key_hue - hue_range < 0:
            # 色相の下限が0未満の場合

            # 下限から180までのマスク
            mask1 = cv2.inRange(hue_img, chroma_key_hue - hue_range + 180, 180)

            # 0から上限までのマスク
            mask2 = cv2.inRange(hue_img, 0, chroma_key_hue + hue_range)

            # 2つのマスクの和
            mask  = cv2.bitwise_or(mask1, mask2)
        else:
            # 0 <= 下限 < 上限 <= 180の場合

            # 色相の下限から上限までのマスク
            mask = cv2.inRange(hue_img, chroma_key_hue - hue_range, chroma_key_hue + hue_range)

        # マスク以外が255で、マスク部分を0にする。
        bin_img = np.where(mask == 0, 255, 0).astype(np.uint8)

    else:
        bin_img = np.zeros(frame.shape[:2], dtype=np.uint8)


    # 二値画像を表示する。
    show_image(window['-image2-'], bin_img)

    # 二値画像から輪郭とマスク画像を得る。
    msg, contour, mask_img = getContour(bin_img)

    # メッセージを表示する。
    window['-msg-'].update(value=msg)

    if msg != '':
        # エラーの場合

        # 黒画面を表示する。
        black_img = np.zeros(frame.shape, dtype=np.uint8)
        show_image(window['-image3-'], black_img)
        return

    # mask_imgはグレースケール画像なので、カラー画像に変換する。
    mask_img = mask_img[:, :, np.newaxis]
    mask_img = np.broadcast_to(mask_img, frame.shape)

    # マスクがオフの画素は白で、オンの画素は原画の色を使う。
    compo_img = np.where(mask_img == 0, white_img, frame)

    if writer is not None:
        # 動画撮影中の場合

        # 画像を1フレーム分として書き込み
        writer.write(compo_img)

        # 動画ファイルへの書き込み件数
        write_cnt += 1

    # 合成画像を表示する。
    show_image(window['-image3-'], compo_img)

def get_chroma_key():
    """画像の中心の色を返す。

    Returns: 画像の中心のBGRとHSV
    """
    # HSV画像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 画像の高さと幅
    h, w = frame.shape[:2]

    # 画像の中心の座標
    cy = h // 2
    cx = w // 2

    # 画像の中心の色相を返す。
    return frame[cy, cx, :], hsv[cy, cx, :]


if __name__ == '__main__':

    # 動画/静止画を保存するフォルダを作る。
    os.makedirs('./capture', exist_ok=True)

    sg.popup_quick_message('パソコンに接続されたカメラを調べています．．．', font='Any 20', background_color='blue')

    # パソコンに接続されたカメラの数
    num_cameras = get_num_cameras()

    if num_cameras == 0:
        # カメラがない場合

        print('camera is not connexted')
        sys.exit(0)

    # カメラの名前のリスト = [ 'カメラ 1', 'カメラ 2', 'カメラ 3', ・・・  ]
    camera_name_list = [ f'カメラ {i+1}' for i in range(num_cameras) ]

    dsp_size = 360

    layout = [
        [
            sg.Column([
                [ 
                    sg.Frame('原画',[
                        [
                            sg.Image(filename='', size=(dsp_size,dsp_size), key='-image1-')
                        ]
                    ])
                ]
                ,
                [ 
                    sg.Text('カメラ', size=(12,1), pad=((10,0),(20,20)) ), 
                    sg.Combo(camera_name_list, default_value=camera_name_list[0], size=(8,1), enable_events=True, readonly=True, key='-camera-')
                ]
                ,
                [ 
                    sg.Text('画像の明るさ', size=(12,1), pad=((10,0),(20,20)) ), 
                    sg.Spin(list(range(-1, 255 + 1)), initial_value=0, size=(8, 1), key='-brightness-', enable_events=True )
                ]
                ,
                [
                    sg.Checkbox('画像を正方形にする。', default=make_square, key='-square-', enable_events=True, pad=((10,0),(20,20)) )
                ]
                ,
                [
                    sg.TabGroup ([
                        [
                            sg.Tab('明度' ,[
                                [
                                    sg.Text('明度の閾値', size=(12,1), pad=((10,0),(20,20)) ), 
                                    sg.Spin(list(range(0, 255 + 1)), initial_value=v_min, size=(8, 1), key='-V-min-', enable_events=True )
                                ]
                            ])
                            ,
                            sg.Tab('色相' ,[
                                [
                                    sg.Button('色相を指定', key='-chroma-key-'),
                                    sg.Text('', size=(2,1), key='-color-sample-', relief=sg.RELIEF_SUNKEN)
                                ]
                                ,
                                [
                                    sg.Text('色相の範囲', size=(12,1), pad=((10,0),(20,20)) ), 
                                    sg.Spin(list(range(0, 20)), initial_value=hue_range, size=(8, 1), key='-hue-range-', enable_events=True )
                                ]
                            ])
                        ]
                    ], key='-bin-method-',  enable_events=True)
                ]
                ,
                [sg.VPush()]
                ,
                [
                    sg.Button('動画撮影', key='-record/pause-', size=(8,1), pad=((10,0),(20,20)) ),
                    sg.Button('写真撮影', key='-shoot-'),
                    sg.Push(),
                    sg.Button('閉じる', key='-close-')
                ]
            ], vertical_alignment='top', expand_y=True)
            ,
            sg.Column([
                [ 
                    sg.Frame('二値画像',[
                        [
                            sg.Image(filename='', size=(dsp_size,dsp_size), key='-image2-')
                        ]
                    ])
                ]
                ,
                [ 
                    sg.Frame('物体',[
                        [
                            sg.Image(filename='', size=(dsp_size,dsp_size), key='-image3-')
                        ]
                    ])
                ]
            ])
        ]
        ,
        [
            sg.Text('', key='-msg-', expand_x=True, relief=sg.RELIEF_SUNKEN ), 
        ]
    ]

    # メインウィンドウ
    window = sg.Window('カメラ', layout, font='Any 16')

    while True:

        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED or event == '-close-':
            # 閉じるボタン

            # カメラのキャプチャ オブジェクトを解放する。
            cap.release()
            break

        elif event == '-camera-':
            # カメラの選択のコンボボックス

            # カメラのキャプチャ オブジェクトを解放する。
            cap.release()

            # コンボボックスの選択されたカメラの名前
            camera_name = values[event]

            # カメラのインデックス
            camera_idx = camera_name_list.index(camera_name)

            # カメラの初期処理をする。
            init_camera(camera_idx)

        elif event == '-record/pause-':
            # 動画撮影/停止ボタン

            if writer is None:
                # 動画撮影中でない場合

                # ボタンのテキストを"停止"にする。
                window['-record/pause-'].update(text='停止')

                window['-square-'].update(disabled = True)

                initWriter()

            else:
                # 動画撮影中の場合

                # ボタンのテキストを"動画撮影"にする。
                window['-record/pause-'].update(text='動画撮影')

                window['-square-'].update(disabled = False)

                # 動画ファイルへのライターを解放する。
                writer.release()
                writer = None

                if not os.path.isfile(write_path):
                    # 動画ファイルへ書けなかった場合

                    print("=" * 80 + '\n\t\tOpenCVのビルド情報\n' + "=" * 80)
                    print(cv2.getBuildInformation())

                    sg.popup('動画ファイルを保存できません。\n\nOpenCVのビルド情報を確認してください。')

                elif write_cnt == 0:
                    # 動画ファイルへの書き込み件数が0の場合

                    sg.popup('物体の画像がありません。')

                    # 動画ファイルを削除する。
                    os.remove(write_path)

                else:
                    sg.popup_quick_message(f'動画を保存しました。\n\n{write_path}', font='Any 20', background_color='blue')

        elif event == '-bin-method-':
            # 明度または色相のタブ

            bin_method = values[event]

        elif event == '-V-min-':
            # 明度の閾値(最小値)
            v_min = int(values[event])

        elif event == '-hue-range-':
            # 色相の範囲
            hue_range = int(values[event])

        elif event == '-brightness-':
            # 画像の明るさ

            brightness = int(values[event])

            # 画像の明るさを設定する。
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

        elif event == '-square-':
            # 画像を正方形にする。

            make_square = values['-square-']

        elif event == '-shoot-':
            # 写真撮影ボタン

            # 静止画をファイルに書き込む。
            now = datetime.datetime.now()
            img_path = f'./capture/{now.strftime("%Y-%m-%d-%H-%M-%S")}.jpg'
            cv2.imwrite(img_path, frame)

            sg.popup_quick_message(f'画像を保存しました。\n\n{img_path}', font='Any 20', background_color='blue')

        elif event == '-chroma-key-':
            # 色相を指定ボタン

            # 画像の中心の色
            bgr, hsv = get_chroma_key()

            # 背景の色相
            chroma_key_hue = int(hsv[0])

            # 背景色を表示
            bg_color = '#%02X%02X%02X' % (bgr[2], bgr[1], bgr[0])
            window['-color-sample-'].update(background_color=bg_color)

        elif event == '__TIMEOUT__':
            if cap is None:
                # 最初の場合

                # カメラの初期処理をする。
                init_camera(0)

            # カメラから1フレームを読む。
            read_one_frame()
