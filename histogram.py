import os
import glob
from PySimpleGUI.PySimpleGUI import Button, Column
import cv2
import sys
import numpy as np
from operator import itemgetter
import PySimpleGUI as sg
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

if __name__ == '__main__':
    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)    
    if not cap.isOpened():

        print(f'can not open video:{video_path}')
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hist = cv2.calcHist([hsv],[0],None,[256],[0,256])

        # for i, (name, col,hi) in enumerate(zip(['H','S','V'], ['r', 'g', 'b'],[180,255,255])):
        for i, (name, col,hi) in enumerate(zip(['H'], ['r'],[180])):
            histr = cv2.calcHist([hsv],[i],None,[180],[0,hi])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.show()        

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()