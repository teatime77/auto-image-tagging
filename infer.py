import time
import glob
import numpy as np
import cv2
import onnx
import onnxruntime as ort

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# print(onnx.helper.printable_graph(onnx_model.graph))

print(ort.get_available_providers())

options = ort.SessionOptions()
options.enable_mem_pattern = False

ort_sess = ort.InferenceSession('model.onnx', sess_options=options, providers=['DmlExecutionProvider'])

ii = ort_sess.get_inputs()
for i in ii:
    print(str(i))
print("")

oo = ort_sess.get_outputs()
for o in oo:
    print(str(o))

cv2.namedWindow('window')

img_dir = '../data/ichigo/img'
for img_path in glob.glob(f'{img_dir}/*.png'):
    bmp = cv2.imread(img_path)

    img = bmp.astype(np.float32)
    img = img / 255.0

    img = cv2.resize(img, dsize=(1280,1280))

    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    start_time = time.time()
    X_img = ort.OrtValue.ortvalue_from_numpy(img)
    ret = ort_sess.run(None, { 'input_1': X_img } )
    sec = '%.1f' % (time.time() - start_time)

    # print(type(img), img.shape, img.dtype, type(ret))
    scores = ret[:5]
    boxes  = ret[5:]
    max_score = 0
    max_score_idx = 0
    for score_idx, score in enumerate(scores):
        idx = np.unravel_index(score.argmax(), score.shape)
        if max_score < score[idx]:
            max_score = score[idx]
            max_score_idx = score_idx
            max_idx = idx
    b, a, row, col = max_idx
    a1 = a * 6
    a2 = (a + 1) * 6
    score = scores[max_score_idx]
    box = boxes[max_score_idx]
    print("    ", sec, scores[max_score_idx].shape, box.shape, max_idx, max_score, score[max_idx], box[0, a1:a2, row, col])

    bmp_h = bmp.shape[0]
    bmp_w = bmp.shape[1]

    num_box_h = box.shape[2]
    num_box_w = box.shape[3]

    box_h = bmp_h / num_box_h
    box_w = bmp_w / num_box_w

    cy = int(row * box_h + box_h / 2) 
    cx = int(col * box_w + box_w / 2)

    cv2.circle(bmp, (int(cx), int(cy)), 10, (255,255,255), -1)

    cv2.imshow('window', bmp)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
    print('next')
