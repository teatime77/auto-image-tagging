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

data_size = 3000
playing = False
saveAll = False
isSaving = False
csvFile = None
AnnoObj = None

classIdx = 0
imageClasses = []

V_lo = 253
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

def show_image(key, img):
    img = cv2.resize(img, dsize=(256, 256))       

    if len(img.shape) == 3:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        # print('image_rgb:type', type(image_rgb), image_rgb.shape, image_rgb.dtype)
        image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
    else:
        image_pil = Image.fromarray(img) # RGBからPILフォーマットへ変換

    image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換

    window[key].update(data=image_tk, size=(256,256))

def stopSave():
    global saveVideoIdx, csvFile, classIdx

    saveVideoIdx = 0
    classIdx = 0

    if csvFile is not None:
        csvFile.close()
        csvFile = None

        with open(f'{output_dir}/train.json', 'w') as f:
            json.dump(AnnoObj, f, indent=4)

    setPlaying(False)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def readCap():
    global cap, saveVideoIdx, csvFile, classIdx

    ret, frame = cap.read()
    if ret:

        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        window['-img-pos-'].update(value=pos)

        showVideo(frame)

        if AnnoObj is not None:
            images_len = len(AnnoObj["images"])

            if data_size <= images_len:

                cap.release()
                stopSave()
                print("保存終了")

            elif images_len % 10 == 0:
                print("images len", images_len)

    else:

        saveVideoIdx += 1

        cap.release()

        if saveVideoIdx < len(imageClasses[classIdx].videoPathes):
            # 同じクラスの別の動画ファイルがある場合

            cap = initCap()
        else:
            # 同じクラスの別の動画ファイルがない場合

            classIdx = (classIdx + 1) % len(imageClasses)
            saveVideoIdx = 0

            if not saveAll or data_size < len(AnnoObj["images"]):

                stopSave()

            else:

                cap = initCap()
        

def showRot(img, cx, cy, w, h, theta_deg):
    global dX, dY, dW, dH, dT

    assert(abs(theta_deg) <= 45)

    # 矩形の左上のXY座標
    min_x, min_y = [ cx - 0.5 * w, cy - 0.5 * h ]

    # 矩形の4頂点の座標
    corners_org = np.array([ [ min_x, min_y ], [ min_x + w, min_y ], [ min_x + w, min_y + h ], [ min_x, min_y + h ]  ])

    # 矩形の中心のXY座標
    center = np.array([cx, cy])

    theta = - math.radians(theta_deg)

    # 回転行列
    rotation = np.array([ [ np.cos(theta), -np.sin(theta) ],
                          [ np.sin(theta),  np.cos(theta) ] ])

    # 矩形の中心が原点(0,0)に来るように平行移動してから、回転をして、元の位置に平行移動する。
    corners_rot = np.matmul(corners_org - center, rotation) + center

    corners_rot = np.int0(corners_rot)

    bbox = _corners2rotatedbbox(corners_rot)
    x2, y2, w2, h2, th2 = bbox

    dx = abs(min_x - x2)
    dy = abs(min_y - y2)
    dw = abs(w - w2)
    dh = abs(h - h2)
    dt = abs(theta_deg - math.degrees(th2))

    if dX < dx or dY < dy or dW < dw or dH < dh or dT < dt:
        dX, dY, dW, dH, dT = [ max(dX,dx), max(dY,dy), max(dW,dw), max(dH,dh), max(dT,dt) ]

    # print(f'    Dx:%.1f y:%.1f w:%.1f h:%.1f th:%.1f' % (dX, dY, dW, dH, dT))
    # print(f'     x:%.1f y:%.1f w:%.1f h:%.1f th:%.1f' % (min_x, min_y, w, h, theta_deg))

    # print(f'    x:%.1f y:%.1f w:%.1f h:%.1f th:%.1f' % (x2, y2, w2, h2, math.degrees(th2)))

    return (bbox, corners_org, corners_rot)

def box_slope(box):
    rad45 = math.radians(45)

    for i1 in range(4):
        i2 = (i1 + 1) % 4

        dx = box[i2][0] - box[i1][0]
        dy = box[i2][1] - box[i1][1]

        theta = math.atan2(dy, dx)
        if abs(theta) <= rad45:
            box2 = []
            for j in range(4):
                k = (i1 + j) % 4
                box2.append(box[k])
            return theta, box2

    return None

def warp_box(box, M):
    box2 = []
    for p in box:
        v1 = np.array([p[0], p[1], 1])
        v2 = np.dot(M, v1)

        box2.append( np.array([v2[0], v2[1]])  )

    return box2


def rotate_box(box, theta):

    center = np.mean(np.array(box), 0)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])

    return [ np.dot(rot, np.array(v) - center) + center for v in box ]



def showContours(frame, bin_img, contours, hierarchy, idx, nest):
    contour = contours[idx]

    # 二値画像の幅と高さ
    width  = bin_img.shape[0]
    height = bin_img.shape[1]

    # 二値画像の面積
    img_area = width * height

    # 輪郭の面積
    area = cv2.contourArea(contour)
    ratio = 100 * np.sqrt(area) / np.sqrt(img_area)

    if ratio < 40:
        return 9 * [None]

    # 輪郭のモーメントを計算する。
    M = cv2.moments(contour)

    # モーメントから重心のXY座標を計算す。
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    rx = int(100 * cx / width)
    ry = int(100 * cy / height)

    if 35 <= rx and rx <= 65 and 35 <= ry and ry <= 65:
        pass
    else:
        print("%s %d [%d, %d, %d, %d] area:%.1f c(%d, %d)" % (' ' * (nest * 4), idx, hierarchy[0][idx][Next_Sibling], hierarchy[0][idx][Previous_Sibling], hierarchy[0][idx][First_Child], hierarchy[0][idx][Parent], ratio, rx, ry))
        return 9 * [None]

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

    # 回転を考慮した外接矩形を得る。
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    cv2.drawContours(clip_img, [ np.int0(box) ], 0, (0,255,0), 2)

    # 矩形の中心、サイズ、回転角を得る。
    ((cx, cy), (w, h), theta_deg) = rect

    if 45 < theta_deg:
        w, h = h, w
        theta_deg -= 90

    # bbox, corners_org, corners_rot = showRot(clip_img, cx, cy, w, h, theta_deg)

    # draw_auxiliary_line(clip_img, cx, cy, corners_org, corners_rot)

    show_image('-image21-', clip_img)

    return contour, mask_img, edge_img, box, cx, cy, w, h, theta_deg

def _corners2rotatedbbox(corners):
    corners = np.array(corners)
    center = np.mean(np.array(corners), 0)
    theta = calc_bearing(corners[0], corners[1])
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    out_points = np.matmul(corners - center, rotation) + center
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

def draw_auxiliary_line(img, cx, cy, corners_org, corners_rot):

    # 矩形の中心に円を描く。
    cv2.circle(img, (int(cx), int(cy)), 10, (255,255,255), -1)

    # 回転前の矩形を緑色で描く。
    cv2.rectangle(img, np.int0(corners_org[0,:]), np.int0(corners_org[2,:]), (0,255,0), 3)

    # 回転後の矩形を緑色で描く。
    cv2.drawContours(img, [ corners_rot ], 0, (0,0,255),2)

    # 各頂点の色 = [ 白, 灰色, 緑, 赤 ]
    point_colors = ((255,255,255), (128,128,128), (0,255,0), (0,0,255))

    # 頂点を中心とする円を描く。
    for i in range(4):
        cv2.circle(img, (int(corners_rot[i,0]), int(corners_rot[i,1])), 10, point_colors[i], -1)

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
    global bgImgPaths, bgImgIdx

    # 原画を表示する。
    show_image('-image11-', frame)

    # グレー画像を表示する。
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    show_image('-image12-', gray_img)

    # 二値画像を表示する。
    bin_img = 255 - cv2.inRange(gray_img, V_lo, V_hi)
    show_image('-image13-', bin_img)


    # 二値化画像から輪郭のリストを得る。
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # RETR_EXTERNAL  RETR_CCOMP 

    assert(len(hierarchy.shape) == 3 and hierarchy.shape[0] == 1 and hierarchy.shape[2] == 4)
    assert(len(contours) == hierarchy.shape[1])

    # print('-' * 50)
    contour = None
    for idx, _ in enumerate(contours):
        if hierarchy[0][idx][Parent] == -1:
            # トップレベルの場合

            contour, mask_img, edge_img, box, cx, cy, w, h, theta_deg = showContours(frame, bin_img, contours, hierarchy, idx, 0)
            if contour is not None:
                break

    if contour is None:
        return

    frame2 = frame.copy()

    # # コントラストと輝度を変える。
    # img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    # img_hsv2 = np.copy(img_hsv)

    # s_mag = random.uniform(0.5, 1.0)
    # v_mag = random.uniform(0.5, 1.0)

    # img_hsv2[:,:,(1)] = img_hsv2[:,:,(1)] * s_mag
    # img_hsv2[:,:,(2)] = img_hsv2[:,:,(2)] * v_mag

    # frame2 = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2BGR)

    # 元画像にマスクをかける。
    clip_img = frame2 * mask_img

    minx = cx - 0.5 * w
    miny = cy - 0.5 * h

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


    # 乱数で回転量を決める。
    angle = random.uniform(-180, 180)
    # print('rot', math.degrees(angle))

    # 回転とスケール
    m1 = cv2.getRotationMatrix2D(center, angle, scale)
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


    box_rot = warp_box(box, M)

    slope, box_rot = box_slope(box_rot)
    if slope is None:
        print('slope is None')
        
        return

    rect_box = rotate_box(box_rot, -slope)

    x, y = rect_box[0]
    w, h = list(rect_box[2] - rect_box[0])

    if True:

        bbox = _corners2rotatedbbox(box_rot)
        x, y, w, h, theta = bbox
        cv2.rectangle(dst_img2, (int(x),int(y)), (int(x+w),int(y+h)), (0,0,255), 3)
    else:
        bbox = [ x, y, w, h, slope ]
        cv2.drawContours(dst_img2, [ np.int0(rect_box) ], 0, (0,0,255), 2)

    cv2.drawContours(dst_img2, [ np.int0(box_rot)  ], 0, (0,255,0), 2)

    cv2.circle(dst_img2, (int(x), int(y)), 10, (255,255,255), -1)

    # 矩形を描く。
    # bbox, corners_org, corners_rot = showRot(dst_img2 , cx + dx, cy + dy, w, h, theta_deg - angle)

    # draw_auxiliary_line(dst_img2, cx + dx, cy + dy, corners_org, corners_rot)

    show_image('-image22-', dst_img2)

    show_image('-image23-', compo_img)

    if csvFile is not None:

        image_id = len(AnnoObj["images"]) + 1

        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        test_img_path = f'{output_dir}/img/{classIdx}-{saveVideoIdx}-{pos}-{image_id}.png'
        cv2.imwrite(test_img_path, compo_img)

        file_name = os.path.basename(test_img_path)
        # csvFile.write(f'{file_name},{xmin},{ymin},{xmax},{ymax},{classIdx+1}\n')

        AnnoObj["images"].append({
            "id" : image_id,
            "width": cols,
            "height": rows,
            "file_name" : file_name            
        })

        anno_id = len(AnnoObj["annotations"]) + 1
        AnnoObj["annotations"].append({
            "id" : anno_id,
            "image_id" : image_id, 
            "category_id" : classIdx + 1,
            "bbox" : bbox ,  # all floats
            "segmentation" : [ [p[0], p[1]] for p in box_rot ] ,
            "area": bbox[2] * bbox[3],           # w * h. Required for validation scores
            "iscrowd": 0            # Required for validation scores            
        })

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

    video_dir = sys.argv[1]
    bg_img_dir = sys.argv[2]

    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/img', exist_ok=True)


    bgImgPaths = [ x for x in glob.glob(f'{bg_img_dir}/*') if os.path.splitext(x)[1] in [ '.jpg', '.png' ] ]

    print(f'背景画像数:{len(bgImgPaths)}')

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
        [ sg.Input(str(data_size), key='-data-size-', size=(6,1)) ]
        ,
        spin('V lo', '-Vlo-', V_lo, 0, 255),
        spin('S mag', '-S_mag-', 100, 10, 200) + spin('V mag', '-V_mag-', 100, 10, 200),
        [ sg.Button('Play', key='-play/pause-'), sg.Button('Save', key='-save-'), sg.Button('Save All', key='-save-all-'), sg.Button('Close')] ]

    # Create the Window
    window = sg.Window('Window Title', layout)

    # tree = window['-tree-']
    # tree.add_treeview_data(node)

    # Event Loop to process "events" and get the "values" of the inputs


    cap = None
    while True:
        # event, values = window.read()
        event, values = window.read(timeout=1)

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

        elif event == '-Vlo-':
            V_lo = int(values[event])

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
            data_size = int(values['-data-size-'])
            class_dir = values['-tree-'][0]
            print("save video dir", class_dir)
            v = [ (i, c) for i, c in enumerate(imageClasses) if class_dir == c.classDir ]
            assert len(v) == 1

            classIdx = v[0][0]
            saveAll = False
            isSaving = True
            AnnoObj = initJson([ imageClasses[classIdx] ])

            saveImgs()

        elif event == '-save-all-':
            data_size = int(values['-data-size-'])
            classIdx = 0;
            saveAll = True
            isSaving = True
            AnnoObj = initJson(imageClasses)

            saveImgs()

        else:

            print('You entered ', event)

    window.close()

    if csvFile is not None:
        csvFile.close()
