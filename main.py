import os
import math
import glob
from PySimpleGUI.PySimpleGUI import Button, Column
import cv2
import sys
import numpy as np
from operator import itemgetter
import PySimpleGUI as sg
from PIL import Image, ImageTk
import random
import time
import json

Next_Sibling, Previous_Sibling, First_Child, Parent = (0, 1, 2, 3)


window_name = 'frame'
bin_name = 'bin'
playing = False
RedBG=False
saveAll = False
isSaving = False
csvFile = None
multipleIdx = 0
framePrev = None

classIdx = 0
imageClasses = []

H_lo =  80
H_hi = 140
S_lo =   0
S_hi = 255
V_lo = 240
V_hi = 255

S_mag =  100
V_mag =  100

dX = 0
dY = 0
dW = 0
dH = 0
dT = 0

class ImageClass:
    def __init__(self, name, class_dir):
        self.name = name
        self.classDir = class_dir
        self.videoPathes = []

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
    global saveVideoIdx

    video_path = imageClasses[classIdx].videoPathes[saveVideoIdx]

    cap = cv2.VideoCapture(video_path)    

    if not cap.isOpened():
        print("動画再生エラー")
        sys.exit()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window['-img-pos-'].update(range=(0, frame_count))
    window['-img-pos-'].update(value=0)
    print(f'再生開始 フレーム数:{cap.get(cv2.CAP_PROP_FRAME_COUNT)} {os.path.basename(video_path)}')

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
    global cap, saveVideoIdx, csvFile, classIdx, framePrev

    if isSaving and multipleIdx != 0:
        showVideo(framePrev)
        return

    ret, frame = cap.read()
    if ret:
        # showImg('-image11-', frame)

        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        window['-img-pos-'].update(value=pos)

        showVideo(frame)
        framePrev = frame

    else:

        saveVideoIdx += 1

        if saveVideoIdx < len(imageClasses[classIdx].videoPathes):
            cap.release()

            cap = initCap()
        else:

            classIdx += 1
            saveVideoIdx = 0

            if saveAll and classIdx < len(imageClasses):
                cap.release()

                cap = initCap()

            else:
                if csvFile is not None:
                    csvFile.close()
                    csvFile = None

                    with open(f'{output_dir}/train.json', 'w') as f:
                        json.dump(AnnoObj, f, indent=4)

                setPlaying(False)
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
def diffHue(hue):
    if hue < H_lo:
        return 90 + hue - H_lo
    elif H_hi < hue:
        return 90 + hue - H_hi
    else:
        return 90

def useHSV(img_hsv):

    # Hueを取り出す。
    hue = img_hsv[:, :, 0]

    # Hueの指定範囲からの差をグレースケールで表示する。
    diff_img = np.where(hue < H_lo, 90 + hue - H_lo, np.where(H_hi < hue, 90 + hue - H_hi, 90))

def showRot(img, cx, cy, w, h, theta_deg, img_key):
    global dX, dY, dW, dH, dT

    assert(abs(theta_deg) <= 45)

    # cx, cy, w, h = np.int0([cx, cy, w, h])
    cv2.circle(img, (int(cx), int(cy)), 10, (255,255,255), -1)

    minx, miny = [ cx - 0.5 * w, cy - 0.5 * h ]
    corners = np.array([ [ minx, miny ], [ minx + w, miny ], [ minx + w, miny + h ], [ minx, miny + h ]  ])
    centre = np.array([cx, cy])

    cv2.rectangle(img, np.int0(corners[0,:]), np.int0(corners[2,:]), (0,255,0),3)


    theta = - math.radians(theta_deg)
    rotation = np.array([ [ np.cos(theta), -np.sin(theta) ],
                          [ np.sin(theta),  np.cos(theta) ] ])

    corners = np.matmul(corners - centre, rotation) + centre
    corners = np.int0(corners)
    cv2.drawContours(img,[ corners ], 0, (0,0,255),2)

    bbox = _corners2rotatedbbox(corners)
    x2, y2, w2, h2, th2 = bbox

    dx = abs(minx - x2)
    dy = abs(miny - y2)
    dw = abs(w - w2)
    dh = abs(h - h2)
    dt = abs(theta_deg - math.degrees(th2))

    if dX < dx or dY < dy or dW < dw or dH < dh or dT < dt:
        dX, dY, dW, dH, dT = [ max(dX,dx), max(dY,dy), max(dW,dw), max(dH,dh), max(dT,dt) ]

    print(f'    Dx:%.1f y:%.1f w:%.1f h:%.1f th:%.1f' % (dX, dY, dW, dH, dT))
    print(f'     x:%.1f y:%.1f w:%.1f h:%.1f th:%.1f' % (minx, miny, w, h, theta_deg))
    # print(f'    x:%.1f y:%.1f w:%.1f h:%.1f th:%.1f' % (x2, y2, w2, h2, math.degrees(th2)))

    cs = ((255,255,255), (128,128, 128), (0,255,0), (0,0,255))
    for i in range(4):
        cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 10, cs[i], -1)

    showImg(img_key, img)

    return (bbox, corners)

def showContours(frame, bin_img, contours, hierarchy, idx, nest):
    contour = contours[idx]

    img_area = bin_img.shape[0] * bin_img.shape[1]
    area = cv2.contourArea(contour)
    ratio = 100 * np.sqrt(area) / np.sqrt(img_area)

    if ratio < 40:
        return 8 * [None]

    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    width  = bin_img.shape[0]
    height  = bin_img.shape[1]
    rx = int(100 * cx / width)
    ry = int(100 * cy / height)

    if 35 <= rx and rx <= 65 and 35 <= ry and ry <= 65:
        pass
    else:
        print("%s %d [%d, %d, %d, %d] area:%.1f c(%d, %d)" % (' ' * (nest * 4), idx, hierarchy[0][idx][Next_Sibling], hierarchy[0][idx][Previous_Sibling], hierarchy[0][idx][First_Child], hierarchy[0][idx][Parent], ratio, rx, ry))
        return 8 * [None]

    # showImg('-image21-', cmp_img)

    conts = [ contour ]

    # 最初の子
    i = hierarchy[0][idx][First_Child]
    while i != -1:
        c = contours[i]
        conts.append(c)

        # 次の兄弟
        i = hierarchy[0][i][Next_Sibling]

    # 輪郭から0と1の二値の内部のマスク画像を作る。
    mask_img = np.zeros(frame.shape, dtype=np.uint8)
    cv2.drawContours(mask_img, conts, -1, (1,1,1), -1)

    # 輪郭から0と1の二値の縁のマスク画像を作る。
    edge_img = np.zeros(frame.shape, dtype=np.uint8)
    cv2.drawContours(edge_img, conts, -1, (1,1,1), 5)

    clip_img = frame * mask_img

    cv2.drawContours(clip_img, conts, -1, (255,0,0), 10)

    # x1, y1, w1, h1 = cv2.boundingRect(mask_img[:,:,0])
    rect = cv2.minAreaRect(contour)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(clip_img,[box],0,(0,0,255),2)



    # 矩形の中心
    ((cx, cy), (w, h), theta_deg) = rect

    if 45 < theta_deg:
        w, h = h, w
        theta_deg -= 90

    showRot(clip_img, cx, cy, w, h, theta_deg, '-image21-')

    return contour, mask_img, edge_img, cx, cy, w, h, theta_deg

def _corners2rotatedbbox(corners):
    corners = np.array(corners)
    centre = np.mean(np.array(corners), 0)
    theta = calc_bearing(corners[0], corners[1])
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    out_points = np.matmul(corners - centre, rotation) + centre
    x, y = list(out_points[0,:])
    w, h = list(out_points[2, :] - out_points[0, :])
    return [x, y, w, h, theta]

def calc_bearing(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    theta = math.atan2(y2 - y1, x2 - x1)
    theta = nor_theta(theta)
    return theta

def nor_theta(theta):
    if theta > math.radians(45):
        theta -= math.radians(90)
        theta = nor_theta(theta)
    elif theta <= math.radians(-45):
        theta += math.radians(90)
        theta = nor_theta(theta)
    return theta

def initJson(image_classes):
    jobj = {
        "annotations":[],
        "images":[],
        "categories":[]
    }

    for i, img_class in enumerate(image_classes):
        o = {
            "supercategory": f'super-{img_class.name}',
            "id": i + 1,
            "name": img_class.name
        }

        jobj["categories"].append(o)

    return jobj;

def showVideo(frame):
    global bgImgPaths, bgImgIdx, multipleIdx

    # 原画を表示する。
    showImg('-image11-', frame)

    # グレー画像を表示する。
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    showImg('-image12-', gray_img)

    # 二値画像を表示する。
    bin_img = 255 - cv2.inRange(gray_img, V_lo, V_hi)
    showImg('-image13-', bin_img)


    # 二値化画像から輪郭のリストを得る。
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # RETR_EXTERNAL  RETR_CCOMP 

    assert(len(hierarchy.shape) == 3 and hierarchy.shape[0] == 1 and hierarchy.shape[2] == 4)
    assert(len(contours) == hierarchy.shape[1])

    print('-' * 50)
    contour = None
    for idx, _ in enumerate(contours):
        if hierarchy[0][idx][Parent] == -1:
            # トップレベルの場合

            contour, mask_img, edge_img, cx, cy, w, h, theta_deg = showContours(frame, bin_img, contours, hierarchy, idx, 0)
            if contour is not None:
                break

    if contour is None:
        return

    if True:
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        frame2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)        

    else:
        # コントラストと輝度を変える。
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        img_hsv2 = np.copy(img_hsv)

        s_mag = random.uniform(0.5, 1.0)
        v_mag = random.uniform(0.5, 1.0)

        img_hsv2[:,:,(1)] = img_hsv2[:,:,(1)] * s_mag
        img_hsv2[:,:,(2)] = img_hsv2[:,:,(2)] * v_mag

        frame2 = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2BGR)

    # 元画像にマスクをかける。
    clip_img = frame2 * mask_img

    minx = cx - 0.5 * w
    miny = cy - 0.5 * h
    # ((minx, miny), (w, h), th) = cv2.minAreaRect(contour)

    rows, cols = frame.shape[:2]

    # 乱数でスケールを決める。
    max_scale = (0.3 * 720.0) / float(max(w, h))
    min_scale = 0.7 * max_scale
    scale = random.uniform(min_scale, max_scale)

    center = (minx + 0.5 * w, miny + 0.5 * h)

    # スケール変換する。
    minx, miny, w, h =[ round(scale * c) for c in (minx, miny, w, h)]
    
    # 乱数で移動量を決める。
    mg = abs(w - h)
    dx = random.uniform(mg + 0.5 * w - cx, cols - (cx + 0.5 * w) - mg)
    dy = random.uniform(mg + 0.5 * h - cy, rows - (cy + 0.5 * h) - mg)

    # mg < cx + dx < cols - w

    # center = (minx + 0.5 * w, miny + 0.5 * h)

    # 0 <= theta_deg - angle <= 90
    # theta_deg - 90 <= angle <= theta_deg
    # angle = random.uniform(theta_deg - 90, theta_deg)

    # -44 <= theta_deg - angle <= 44
    # theta_deg - 44 <= angle <= theta_deg + 44
    angle = random.uniform(theta_deg - 44, theta_deg + 44)
    assert(-44 <= theta_deg - angle and theta_deg - angle <= 44)

    # 回転とスケール
    m1 = cv2.getRotationMatrix2D(center, angle , scale)
    m1 = np.concatenate((m1, np.array([[0.0, 0.0, 1.0]])))

    # 平行移動
    m2 = np.array([
        [ 1, 0, dx], 
        [ 0, 1, dy], 
        [ 0, 0,  1]
    ], dtype=np.float32)

    m3 = np.dot(m2, m1)
    M = m3[:2,:]

    # 画像に変換行列を作用させる。
    dst_img2 = cv2.warpAffine(clip_img, M, (cols, rows))
    warp_img = cv2.warpAffine(frame2, M, (cols, rows))
    mask_img2 = cv2.warpAffine(mask_img, M, (cols, rows))
    edge_img2 = cv2.warpAffine(edge_img, M, (cols, rows))

    # 背景画像ファイルを読む。
    bg_img = cv2.imread(bgImgPaths[bgImgIdx])
    bgImgIdx = (bgImgIdx + 1) % len(bgImgPaths)

    # 背景画像を元画像と同じサイズにする。
    bg_img = cv2.resize(bg_img, dsize=frame.shape[:2])                    

    # 内部のマスクを使って、背景画像と元画像を合成する。
    compo_img = np.where(mask_img2 == 0, bg_img, warp_img)

    # 縁のマスクを使って、背景画像と元画像を合成する。
    blend_img = cv2.addWeighted(bg_img, 0.7, warp_img, 0.3, 0.0)
    compo_img = np.where(edge_img2 == 0, compo_img, blend_img)

    mask_img3 = cv2.cvtColor(mask_img2, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    x1, y1, w1, h1 = cv2.boundingRect(mask_img3)

    # 矩形の左上と右下の座標
    xmin, ymin, xmax, ymax = ( x1, y1, x1 + w1, y1 + h1 )

    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    test_img_path = f'{output_dir}/img/{classIdx}-{saveVideoIdx}-{pos}-{multipleIdx}.png'
    cv2.imwrite(test_img_path, compo_img)


    # 矩形を描く。
    showRot(dst_img2 , cx + dx, cy + dy, w, h, theta_deg - angle, '-image31-')
    bbox, corners = showRot(compo_img, cx + dx, cy + dy, w, h, theta_deg - angle, '-image32-')


    file_name = os.path.basename(test_img_path)

    if csvFile is not None:
        csvFile.write(f'{file_name},{xmin},{ymin},{xmax},{ymax},{classIdx+1}\n')

        image_id = len(AnnoObj["images"]) + 1
        AnnoObj["images"].append({
            "id" : image_id,
            "file_name" : file_name            
        })

        anno_id = len(AnnoObj["annotations"]) + 1
        AnnoObj["annotations"].append({
            "id" : anno_id,
            "image_id" : image_id, 
            "category_id" : classIdx + 1,
            "bbox" : bbox ,  # all floats
            "area": bbox[2] * bbox[3],           # w * h. Required for validation scores
            "iscrowd": 0            # Required for validation scores            
        })

    multiple = int(values['-multiple-'])
    multipleIdx = (multipleIdx + 1) % multiple

def get_tree_data(video_dir):
    global video_pathes, imageClasses

    video_pathes = []
    imageClasses = []

    treedata = sg.TreeData()

    for class_dir in glob.glob(f'{video_dir}/*'):
        category_name = os.path.basename(class_dir)

        img_class = ImageClass(category_name, class_dir)
        imageClasses.append(img_class)

        treedata.Insert('', class_dir, category_name, values=[])
        print(f'category:{category_name} {str(class_dir)}')

        for video_path in glob.glob(f'{class_dir}/*'):
            video_name = os.path.basename(video_path)

            video_path_str = str(video_path)
            print(f'video:{video_path_str}')        

            treedata.Insert(class_dir, video_path_str, video_name, values=[video_path_str])

            img_class.videoPathes.append(video_path_str)
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
     
def spin(label, key, val, min_val, max_val):
    return [ 
        sg.Text(label, size=(6,1)), sg.Text("", size=(6,1)), 
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(10, 1), key=key, enable_events=True )
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

def saveImgs():
    global cap, saveVideoIdx, csvFile, classIdx

    saveVideoIdx = 0

    for img_path in glob.glob(f'{output_dir}/img/*.png'):
        print(f'削除:{img_path}')
        os.remove(img_path)

    csvFile = open(f'{output_dir}/target.csv', 'w')
    csvFile.write('image,xmin,ymin,xmax,ymax,label\n')

    cap = initCap()

def showImgPos():
    if cap is not None:
        pos = int(values['-img-pos-'])
        print(f'再生位置:{pos}')
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        readCap()

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

    output_dir = sys.argv[3]
    print(f'変換先:{output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/img', exist_ok=True)


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
                [ sg.Image(filename='', size=(256,256), key='-image21-') ],
                [ sg.Image(filename='', size=(256,256), key='-image31-') ]
            ])
            ,
            sg.Column([
                [ sg.Image(filename='', size=(256,256), key='-image12-') ],
                [ sg.Image(filename='', size=(256,256), key='-image22-') ],
                [ sg.Image(filename='', size=(256,256), key='-image32-') ]
            ])
            ,
            sg.Column([
                [ sg.Image(filename='', size=(256,256), key='-image13-') ],
                [ sg.Image(filename='', size=(256,256), key='-image23-') ],
                [ sg.Image(filename='', size=(256,256), key='-image33-') ]
            ])
        ]
        ,
        [ sg.Slider(range=(0,100), default_value=0, size=(100,15), orientation='horizontal', change_submits=True, key='-img-pos-') ]
        ,
        spin('H lo', '-Hlo-', H_lo, 0, 180) + spin('H hi', '-Hhi-', H_hi, 0, 180) + [ sg.Image(filename='', key='-Hbar-'), sg.Checkbox('赤背景', default=RedBG, key='-RedBG-', enable_events=True) ],
        spin('S lo', '-Slo-', S_lo, 0, 255) + spin('S hi', '-Shi-', S_hi, 0, 255) + [ sg.Image(filename='', key='-Sbar-') ],
        spin('V lo', '-Vlo-', V_lo, 0, 255) + spin('V hi', '-Vhi-', V_hi, 0, 255) + [ sg.Image(filename='', key='-Vbar-') ],
        spin('倍数', '-multiple-', 1, 1, 10) + spin('S mag', '-S_mag-', 100, 10, 200) + spin('V mag', '-V_mag-', 100, 10, 200),
        [sg.Text('Enter something on Row 2'), sg.InputText()],
        [ sg.Button('Play', key='-play/pause-'), sg.Button('Save', key='-save-'), sg.Button('Save All', key='-save-all-'), sg.Button('Close')] ]

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

        if event == sg.WIN_CLOSED or event == 'Close': # if user closes window or clicks cancel
            break

        if event == '-tree-':
            print(f'クリック [{values[event]}] [{values[event][0]}]')
            video_path = values[event][0]
            if video_path in video_pathes:

                if cap is not None:
                    cap.release()

                v = [ (i, c) for i, c in enumerate(imageClasses) if video_path in c.videoPathes ]
                assert len(v) == 1

                classIdx = v[0][0]
                img_class = v[0][1]
                saveVideoIdx = img_class.videoPathes.index(video_path)

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

        elif event == '-S_mag-':
            S_mag = int(values[event])
            showImgPos()

        elif event == '-V_mag-':
            V_mag = int(values[event])
            showImgPos()

        elif event == '__TIMEOUT__':
            if cap is not None and playing:
                readCap()

        elif event == '-img-pos-':
            showImgPos()

        elif event == '-play/pause-':
            setPlaying(not playing)

        elif event == '-save-':
            class_dir = values['-tree-'][0]
            print("save video dir", class_dir)
            v = [ (i, c) for i, c in enumerate(imageClasses) if class_dir == c.classDir ]
            assert len(v) == 1

            classIdx = v[0][0]
            saveAll = False
            isSaving = True
            multipleIdx = 0
            multipleIdx = 0
            AnnoObj = initJson([ imageClasses[classIdx] ])

            saveImgs()

        elif event == '-save-all-':
            classIdx = 0;
            saveAll = True
            isSaving = True
            AnnoObj = initJson(imageClasses)

            saveImgs()

        elif event == '-RedBG-':
            RedBG = window[event].get()
            colorBar()

        else:

            print('You entered ', event)

    window.close()

    if csvFile is not None:
        csvFile.close()
