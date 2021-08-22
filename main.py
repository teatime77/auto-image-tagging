import os
import glob
import cv2
import sys
import numpy as np
from operator import itemgetter




delay = 1
window_name = 'frame'
bin_name = 'bin'

H_lo =  65
H_hi = 127
S_lo = 178
S_hi = 255
V_lo =   0
V_hi = 208

def printing(position):
    global H_lo, H_hi, S_lo, S_hi, V_lo, V_hi

    H_lo = cv2.getTrackbarPos('H lo', window_name)
    H_hi = cv2.getTrackbarPos('H hi', window_name)
    S_lo = cv2.getTrackbarPos('S lo', window_name)
    S_hi = cv2.getTrackbarPos('S hi', window_name)
    V_lo = cv2.getTrackbarPos('V lo', window_name)
    V_hi = cv2.getTrackbarPos('V hi', window_name)
    print(H_lo, H_hi, S_lo, S_hi, V_lo, V_hi)








def showVideo(video_path, category_name):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bin_img = cv2.inRange(hsv, (H_lo, S_lo, V_lo), (H_hi, S_hi, V_hi))
        # print(type(bin_img), bin_img.shape)

        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)   # RETR_EXTERNAL RETR_TREE

        # areas = [ cv2.contourArea(x) for x in contours ]
        # print(areas)

        # 輪郭と面積の対のリスト
        contour_areas = [ (x, cv2.contourArea(x)) for x in contours ]

        # 面積が1000以上で全体画像の70%未満の輪郭を抽出
        img_area = frame.shape[0] * frame.shape[1]
        contour_areas = [ x for x in contour_areas if 1000 <= x[1] and x[1] < 0.7 * img_area ]

        if len(contour_areas) == 0:
            continue

        # 面積が大きい順にソート
        # contour_areas = sorted(contour_areas, key=itemgetter(1), reverse=True)

        # 輪郭のリスト        
        contours = [ x[0] for x in contour_areas ]

        print([ x[1] for x in contour_areas ])

        con_img = frame.copy()
        # for idx in range(min(len(contours), len(colors))):
        #     con_img = cv2.drawContours(con_img, contours, idx, colors[idx], 5)
        con_img = cv2.drawContours(con_img, contours, -1, (0,255,0), 5)
        cv2.imshow('contours', con_img)

        # print(type(hsv), hsv.shape)
        cv2.imshow(window_name, frame)
        cv2.imshow(bin_name, bin_img)

        if cv2.waitKey(delay) & 0xFF == ord('q'):

            cap.release()
            return False
                

    cap.release()
    return True

def main(video_dir):
    # video_dir = '/home/hamada/ビデオ/Chroma'

    for category_path in glob.glob(f'{video_dir}/*'):
        category_name = os.path.basename(category_path)

        for video_path in glob.glob(f'{category_path}/*'):
            ok = showVideo(video_path, category_name)
            if not ok:
                return False

if __name__ == '__main__':
    print(cv2.getBuildInformation())

    # 色のテーブルを作る。
    colors = []
    for r in [ 255, 0 ]:
        for g in [ 255, 0]:
            for b in [ 255, 0 ]:
                colors.append((r, g, b))

    colors = colors[1:] + colors[:1]       


    cv2.namedWindow(window_name)
    for name , val in zip([ 'H lo', 'H hi', 'S lo', 'S hi', 'V lo', 'V hi',  ], [ H_lo, H_hi, S_lo, S_hi, V_lo, V_hi ]):
        cv2.createTrackbar(name, window_name, val, 255, printing)    

    video_dir = sys.argv[1]

    ok = True
    while ok:
        ok = main(video_dir)

    cv2.destroyWindow(window_name)    
