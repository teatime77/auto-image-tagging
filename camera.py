import cv2
import PySimpleGUI as sg
from PIL import Image, ImageTk
from util import spin, show_image, getContour

V_lo = 253

def readCap():
    ret, frame = cap.read()

    if frame is None:
        return
        
    h, w, c = frame.shape

    s1 = (h - img_size) // 2
    e1 = s1 + img_size

    s2 = (w - img_size) // 2
    e2 = s2 + img_size

    img = frame[s1:e1, s2:e2, :]

    # 画像を1フレーム分として書き込み
    writer.write(img)

    show_image(window['-image11-'], img)

    # グレー画像を表示する。
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    show_image(window['-image12-'], gray_img)

    # gray_img2 = cv2.equalizeHist(gray_img)
    # show_image(window['-image13-'], gray_img2)

    # 二値画像を表示する。
    bin_img = 255 - cv2.inRange(gray_img, V_lo, 255)
    show_image(window['-image22-'], bin_img)

    # 二値画像を表示する。
    # bin_img2 = 255 - cv2.inRange(gray_img2, V_lo, 255)
    # show_image(window['-image23-'], bin_img2)

    contour, contour_family, mask_img, edge_img = getContour(bin_img)
    if contour is None:
        return

    clip_img = frame * mask_img

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

cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する

brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
exposure   = int(cap.get(cv2.CAP_PROP_EXPOSURE))
contrast   = int(cap.get(cv2.CAP_PROP_CONTRAST))

print('BRIGHTNESS', brightness)
print('EXPOSURE'  , exposure)
print('CONTRAST'  , contrast)
print('FPS'       , cap.get(cv2.CAP_PROP_FPS))
print('AUTO EXPOSURE', cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))

WIDTH  = 960
HEIGHT = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH )) == WIDTH
assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == HEIGHT

WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

img_size   = min(WIDTH, HEIGHT)

print('WIDTH'     , WIDTH)
print('HEIGHT'    , HEIGHT)


frame_rate = cap.get(cv2.CAP_PROP_FPS)

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
writer = cv2.VideoWriter('./outtest.mp4', fmt, frame_rate, (img_size, img_size)) # ライター作成



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
    [ sg.Button('Close') ]
]

window = sg.Window('Window Title', layout)

while True:

    event, values = window.read(timeout=1)

    if event == sg.WIN_CLOSED or event == 'Close':
        break

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



cap.release()

writer.release() # ファイルを閉じる