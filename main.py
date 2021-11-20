import os
import glob
from PySimpleGUI.PySimpleGUI import Button, Column
import cv2
import sys
import numpy as np
from operator import itemgetter
import PySimpleGUI as sg
from PIL import Image, ImageTk
import random


window_name = 'frame'
bin_name = 'bin'
playing = False
RedBG=False
csvFile = None

H_lo =  90
H_hi = 180
S_lo =   0
S_hi = 255
V_lo =   0
V_hi = 255

def printing(position):
    global H_lo, H_hi, S_lo, S_hi, V_lo, V_hi

    H_lo = cv2.getTrackbarPos('H lo', window_name)
    H_hi = cv2.getTrackbarPos('H hi', window_name)
    S_lo = cv2.getTrackbarPos('S lo', window_name)
    S_hi = cv2.getTrackbarPos('S hi', window_name)
    V_lo = cv2.getTrackbarPos('V lo', window_name)
    V_hi = cv2.getTrackbarPos('V hi', window_name)
    print(H_lo, H_hi, S_lo, S_hi, V_lo, V_hi)



def initCap():
    global saveVideoPathes, saveVideoIdx

    video_path = saveVideoPathes[saveVideoIdx]

    for img_path in glob.glob('tmp/*.png'):
        print(img_path)
        os.remove(img_path)

    cap = cv2.VideoCapture(video_path)    

    if not cap.isOpened():
        sys.exit()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window['-img-pos-'].update(range=(0, frame_count))
    window['-img-pos-'].update(value=0)
    print(f'再生開始 フレーム数:{cap.get(cv2.CAP_PROP_FRAME_COUNT)}')

    setPlaying(True)

    return cap

def showImg(key, img):
    img = cv2.resize(img, dsize=(256, 256))       

    if len(img.shape) == 3:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        # print('image_rgb:type', type(image_rgb), image_rgb.shape, image_rgb.dtype)
        image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
    else:
        image_pil = Image.fromarray(img) # RGBからPILフォーマットへ変換

    image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換

    window[key].update(data=image_tk, size=(256,256))

def readCap():
    global cap, saveVideoPathes, saveVideoIdx, csvFile

    ret, frame = cap.read()
    if ret:
        # showImg('-image11-', frame)

        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        window['-img-pos-'].update(value=pos)

        showVideo(frame)

    else:

        saveVideoIdx += 1

        if saveVideoIdx < len(saveVideoPathes):
            cap.release()

            cap = initCap()
        else:

            if csvFile is not None:
                csvFile.close()
                csvFile = None

            setPlaying(False)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
def diffHue(hue):
    if hue < H_lo:
        return 90 + hue - H_lo
    elif H_hi < hue:
        return 90 + hue - H_hi
    else:
        return 90

def showVideo(frame):
    global bgImgPaths, bgImgIdx
    
    # BGRからHSVに変換する。
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSVの範囲を0と255に2値化する。
    bin_img = cv2.inRange(hsv, (H_lo, S_lo, V_lo), (H_hi, S_hi, V_hi))
    assert(bin_img.shape == hsv.shape[:2])

    # 二値化画像から輪郭のリストを得る。
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)   # RETR_EXTERNAL RETR_TREE

    # 輪郭と面積の対のリスト
    contour_areas = [ (x, cv2.contourArea(x)) for x in contours ]

    # 元画像の面積
    img_area = frame.shape[0] * frame.shape[1]

    # 輪郭と面積の平方根の比の対のリスト
    contour_areas_ratio = [ (x[0], int(100 * np.sqrt(x[1]) / np.sqrt(img_area)) ) for x in contour_areas ]
    
    # 面積の平方根が元画像の10%以上で70%未満の輪郭を抽出する。
    contour_areas_ratio = [ x for x in contour_areas_ratio if 10 <= x[1] and x[1] < 70 ]

    if len(contour_areas_ratio) != 1:
        # 条件を満たす輪郭が1つでない場合

        return

    # 対象の輪郭      
    contour = contour_areas_ratio[0][0]

    # 輪郭から0と1の二値の内部のマスク画像を作る。
    mask_img = np.zeros(frame.shape, dtype=np.uint8)
    cv2.drawContours(mask_img, [ contour ], -1, (1,1,1), -1)

    # 輪郭から0と1の二値の縁のマスク画像を作る。
    edge_img = np.zeros(frame.shape, dtype=np.uint8)
    cv2.drawContours(edge_img, [ contour ], -1, (1,1,1), 5)

    # 元画像から
    dst_img = frame * mask_img

    x, y, w, h = cv2.boundingRect(contour)


    # rect_img = np.zeros(frame.shape, dtype=np.uint8)
    cv2.rectangle(dst_img, (x, y), (x+w, y+h), (0, 255, 0), 3)    

    rows, cols = frame.shape[:2]

    # 乱数でスケールを決める。
    scale = random.uniform(0.5, 1)

    # スケール変換する。
    x, y, w, h =[ round(scale * x) for x in (x, y, w, h)]
    
    # 乱数で移動量を決める。
    dx = random.randint(-x, cols - (x + w))
    dy = random.randint(-y, rows - (y + h))

    # スケールと平行移動の変換行列
    M = np.float32([
        [scale,     0, dx],
        [    0, scale, dy]
    ])

    # 画像に変換行列を作用させる。
    dst_img = cv2.warpAffine(dst_img, M, (cols, rows))
    warp_img = cv2.warpAffine(frame, M, (cols, rows))
    mask_img = cv2.warpAffine(mask_img, M, (cols, rows))
    edge_img = cv2.warpAffine(edge_img, M, (cols, rows))

    # 背景画像ファイルを読む。
    bg_img = cv2.imread(bgImgPaths[bgImgIdx])
    bgImgIdx = (bgImgIdx + 1) % len(bgImgPaths)

    # 背景画像を元画像と同じサイズにする。
    bg_img = cv2.resize(bg_img, dsize=frame.shape[:2])                    

    # 内部のマスクを使って、背景画像と元画像を合成する。
    compo_img = np.where(mask_img == 0, bg_img, warp_img)

    # 縁のマスクを使って、背景画像と元画像を合成する。
    blend_img = cv2.addWeighted(bg_img, 0.7, warp_img, 0.3, 0.0)
    compo_img = np.where(edge_img == 0, compo_img, blend_img)

    # 矩形の左上と右下の座標
    xmin, ymin, xmax, ymax = ( dx + x, dy + y, dx + x + w, dy + y + h )

    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    test_img_path = os.path.join(test_img_dir, f'{saveVideoIdx}-{pos}.png')         
    cv2.imwrite(test_img_path, compo_img)

    file_name = os.path.basename(test_img_path)
    csvFile.write(f'{file_name},{xmin},{ymin},{xmax},{ymax}\n')

    # 矩形を描く。
    cv2.rectangle(compo_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)    

    showImg('-image11-', dst_img)
    showImg('-image21-', compo_img)

    # Hueを取り出す。
    hue = hsv[:, :, 0]

    # Hueの指定範囲からの差をグレースケールで表示する。
    diff_img = np.where(hue < H_lo, 90 + hue - H_lo, np.where(H_hi < hue, 90 + hue - H_hi, 90))

    showImg('-image12-', diff_img)

def get_tree_data(video_dir):
    global video_pathes

    video_pathes = []

    treedata = sg.TreeData()

    for category_path in glob.glob(f'{video_dir}/*'):
        category_name = os.path.basename(category_path)

        treedata.Insert('', category_path, category_name, values=[])

        for video_path in glob.glob(f'{category_path}/*'):
            video_name = os.path.basename(video_path)

            video_path_str = str(video_path)
            treedata.Insert(category_path, video_path_str, video_name, values=[video_path_str])

            video_pathes.append(video_path_str)

    return treedata

def colorBar():
    width = 80
    height = 18

    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    for HSV in [ 'H', 'S', 'V' ]:
        for y in range(height):
            for x in range(width):
                if   HSV == 'H':
                    hue = H_lo + (H_hi - H_lo) * x / width
                    if RedBG:
                        hue = (hue + 90) % 180
                    hsv[y, x, 0] = hue
                    hsv[y, x, 1] = 255
                    hsv[y, x, 2] = 255
                else:
                    hsv[y, x, 0] = (H_lo + H_hi) / 2
                    if HSV == 'S':
                        hsv[y, x, 1] = S_lo + (S_hi - S_lo) * x / width
                        hsv[y, x, 2] = 255
                    elif HSV == 'V':                
                        hsv[y, x, 1] = (S_lo + S_hi) / 2
                        hsv[y, x, 2] = V_lo + (V_hi - V_lo) * x / width

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        img_pil = Image.fromarray(rgb, mode='RGB')

        hbar_img = ImageTk.PhotoImage(image=img_pil)
        window[f'-{HSV}bar-'].update(data=hbar_img)
     
def spin(label, val, key):
    if label[0] == 'H':
        max_val = 180
    else:
        max_val = 255

    return [ 
        sg.Text(label, size=(6,1)), sg.Text("", size=(6,1)), 
        sg.Spin(list(range(0, max_val + 1)), initial_value=val, size=(10, 1), key=key, enable_events=True )
    ]

def setPlaying(is_playing):
    global playing

    playing = is_playing
    if playing:
        window['-play/pause-'].update(text='Pause')
        print('show pause')
    else:
        window['-play/pause-'].update(text='Play')
        print('show play')

def saveImgs(save_video_dir):
    global cap, saveVideoPathes, saveVideoIdx, csvFile

    saveVideoPathes = [ x for x in glob.glob(f'{save_video_dir}/*') if x in video_pathes]
    saveVideoIdx = 0

    csvFile = open('test.csv', 'w')
    csvFile.write('image,xmin,ymin,xmax,ymax\n')

    cap = initCap()

if __name__ == '__main__':
    print(cv2.getBuildInformation())

    # 色のテーブルを作る。
    colors = []
    for r in [ 255, 0 ]:
        for g in [ 255, 0]:
            for b in [ 255, 0 ]:
                colors.append((r, g, b))

    colors = colors[1:] + colors[:1]       


    # cv2.namedWindow(window_name)
    # for name , val in zip([ 'H lo', 'H hi', 'S lo', 'S hi', 'V lo', 'V hi',  ], [ H_lo, H_hi, S_lo, S_hi, V_lo, V_hi ]):
    #     cv2.createTrackbar(name, window_name, val, 255, printing)    

    video_dir = sys.argv[1]
    bg_img_dir = sys.argv[2]

    test_img_dir = sys.argv[3]
    print(f'変換先:{test_img_dir}')
    os.makedirs(test_img_dir, exist_ok=True)


    bgImgPaths = [ x for x in glob.glob(f'{bg_img_dir}/*') if os.path.splitext(x)[1] in [ '.jpg', '.png' ] ]
    bgImgIdx = 0

    sg.theme('DarkAmber')   # Add a touch of color

    treedata = get_tree_data(video_dir)


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
                [ sg.Slider(range=(0,100), default_value=0, size=(100,15), orientation='horizontal', change_submits=True, key='-img-pos-') ],
                [ sg.Image(filename='', size=(256,256), key='-image21-') ]
            ])
            ,
            sg.Column([
                [ sg.Image(filename='', size=(256,256), key='-image12-') ],
                [ sg.Image(filename='', size=(256,256), key='-image22-') ]
            ])
                    
        ],
        spin('H lo', H_lo, '-Hlo-') + spin('H hi', H_hi, '-Hhi-') + [ sg.Image(filename='', key='-Hbar-'), sg.Checkbox('赤背景', default=RedBG, key='-RedBG-', enable_events=True) ],
        spin('S lo', S_lo, '-Slo-') + spin('S hi', S_hi, '-Shi-') + [ sg.Image(filename='', key='-Sbar-') ],
        spin('V lo', V_lo, '-Vlo-') + spin('V hi', V_hi, '-Vhi-') + [ sg.Image(filename='', key='-Vbar-') ],
        [sg.Text('Some text on Row 1')],
        [sg.Text('Enter something on Row 2'), sg.InputText()],
        [ sg.Button('Play', key='-play/pause-'), sg.Button('Save', key='-save-'), sg.Button('Cancel')] ]

    # Create the Window
    window = sg.Window('Window Title', layout)

    # tree = window['-tree-']
    # tree.add_treeview_data(node)

    # Event Loop to process "events" and get the "values" of the inputs


    cap = None
    is_first = True
    while True:
        # event, values = window.read()
        event, values = window.read(timeout=1)

        if is_first:
            is_first = False
            colorBar()

        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            break

        if event == '-tree-':
            print(values[event])
            video_path = values[event][0]
            if video_path in video_pathes:

                if cap is not None:
                    cap.release()

                saveVideoPathes = [ video_path ]
                saveVideoIdx = 0

                cap = initCap()

        elif event == '-Hlo-':
            H_lo = int(values[event])
            colorBar()
        elif event == '-Hhi-':
            H_hi = int(values[event])
            colorBar()
        elif event == '-Slo-':
            S_lo = int(values[event])
            colorBar()
        elif event == '-Shi-':
            S_hi = int(values[event])
            colorBar()
        elif event == '-Vlo-':
            V_lo = int(values[event])
            colorBar()
        elif event == '-Vhi-':
            V_hi = int(values[event])
            colorBar()
            
        elif event == '__TIMEOUT__':
            if cap is not None and playing:
                readCap()

        elif event == '-img-pos-':
            if cap is not None:
                pos = int(values['-img-pos-'])
                print(f'再生位置:{pos}')
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                readCap()

        elif event == '-play/pause-':
            setPlaying(not playing)

        elif event == '-save-':
            saveImgs(values['-tree-'][0])

        elif event == '-RedBG-':
            RedBG = window[event].get()
            colorBar()

        else:

            print('You entered ', event)

    window.close()

    if csvFile is not None:
        csvFile.close()
