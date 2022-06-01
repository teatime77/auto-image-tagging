import sys
import glob
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import openvino
from openvino.runtime import Core
from openvino.pyopenvino import ConstOutput
import time
from infer_tool import getBox, getInputImg, receiveBmp, sendBox

ie = Core()

devices = ie.available_devices

for device in devices:
    device_name = ie.get_property(device_name=device, name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

model_file = 'model.xml'
model_file = 'cola_720.xml'
model_file = 'cola_16.xml'
model_file = 'ichigo_16.xml'

img_dir = '../data/cola/img'
img_dir = '../data/ichigo/img'

model = ie.read_model(model=model_file)
compiled_model = ie.compile_model(model=model, device_name="GPU")

input_layer_ir = next(iter(compiled_model.inputs))

# Create inference request
request = compiled_model.create_infer_request()

cv2.namedWindow('window')

reader = receiveBmp()

while True:
    try:
        bmp = reader.__next__()
    except StopIteration:
        break

    img = getInputImg(bmp)

    print('start infer')

    start_time = time.time()
    ret = request.infer({input_layer_ir.any_name: img})
    sec = '%.1f' % (time.time() - start_time)
    
    print(sec, type(ret))

    scores = [None] * 5
    boxes  = [None] * 5

    score_names = [ 'score_1', 'score_2', 'score_3', 'score_4', 'score_5' ]
    box_names = [ 'box_1', 'box_2', 'box_3', 'box_4', 'box_5' ]
    for k, v in ret.items():
        assert type(k) is ConstOutput
        print("    ", type(k), type(k.names), k.names, type(v), v.shape)

        name = k.any_name
        if name in score_names:
            score_idx =  score_names.index(name)
            scores[score_idx] = v

        if name in box_names:
            box_idx = box_names.index(name)
            boxes[box_idx] = v

    cx, cy = getBox(scores, boxes, bmp.shape)

    sendBox(cx, cy)

    cv2.circle(bmp, (int(cx), int(cy)), 10, (255,255,255), -1)

    cv2.imshow('window', bmp)
    k = cv2.waitKey(1)
    # if k == ord('q'):
    #     break
