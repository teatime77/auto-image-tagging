import sys
import time
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from infer_tool import getBox, getInputImg, readBmpFiles, receiveBmp

model_file = '../data/model/ichigo/model.onnx'
img_dir    = '../data/ichigo/img'

onnx_model = onnx.load(model_file)
onnx.checker.check_model(onnx_model)

# print(onnx.helper.printable_graph(onnx_model.graph))

print(ort.get_available_providers())

options = ort.SessionOptions()
options.enable_mem_pattern = False

ort_sess = ort.InferenceSession(model_file, sess_options=options, providers=['DmlExecutionProvider'])

ii = ort_sess.get_inputs()
for i in ii:
    print(str(i))
print("")

oo = ort_sess.get_outputs()
for o in oo:
    print(str(o))

# if len(sys.argv) == 2 and sys.argv[1] == 'net':
# else:


cv2.namedWindow('window')

reader = readBmpFiles(img_dir)

while True:
    try:
        bmp = reader.__next__()
    except StopIteration:
        break

    # for img_path in glob.glob(f'{img_dir}/*.png'):
    #     bmp = cv2.imread(img_path)

    img = getInputImg(bmp)

    print('start infer')

    start_time = time.time()
    X_img = ort.OrtValue.ortvalue_from_numpy(img)
    ret = ort_sess.run(None, { 'input_1': X_img } )
    sec = '%.1f' % (time.time() - start_time)

    print(sec, type(ret))

    scores = ret[:5]
    boxes  = ret[5:]

    cx, cy = getBox(scores, boxes, bmp.shape)

    cv2.circle(bmp, (int(cx), int(cy)), 10, (255,255,255), -1)

    cv2.imshow('window', bmp)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
