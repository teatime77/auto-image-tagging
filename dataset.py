import os
import glob
from PySimpleGUI.PySimpleGUI import Button, Column
import cv2
import sys
import numpy as np
from operator import itemgetter
import PySimpleGUI as sg
from PIL import Image, ImageTk

imgSize = 512

if __name__ == '__main__':

    src_dir = sys.argv[1]
    dst_dir = os.path.join(src_dir, sys.argv[2])

    print(f'変換先:{dst_dir}')
    os.makedirs(dst_dir, exist_ok=True)

    for img_path in glob.glob(f'{src_dir}/*'):
        if os.path.splitext(img_path)[1] in [ '.jpg', '.png' ]:
            img = cv2.imread(img_path)
            height, width, num_channel = img.shape
            if num_channel != 3:
                print(img.shape, img_path)
                continue

            if height < width:
                w = imgSize * width // height
                img = cv2.resize(img, dsize=(w, imgSize))                    

                x = (w - imgSize) // 2
                img = img[ : , x:x+imgSize, : ]

            else:
                h = imgSize * height // width
                img = cv2.resize(img, dsize=(imgSize, h))       

                y = (h - imgSize) // 2
                img = img[ y:y+imgSize , : , :]

            assert(img.shape == (imgSize, imgSize, 3))
            
            dst_path = os.path.join(dst_dir, os.path.basename(img_path))         
            cv2.imwrite(dst_path, img)
            print(dst_path)

