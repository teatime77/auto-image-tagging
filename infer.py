import glob
import numpy as np
import cv2
import onnx
import onnxruntime as ort

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# print(onnx.helper.printable_graph(onnx_model.graph))

ort_sess = ort.InferenceSession('model.onnx')
ii = ort_sess.get_inputs()
for i in ii:
    print(str(i))
print("")

oo = ort_sess.get_outputs()
for o in oo:
    print(str(o))

img_dir = 'C:/usr/prj/data/sweets3/img'
for img_path in glob.glob(f'{img_dir}/*.png'):
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img = img / 255.0

    img = cv2.resize(img, dsize=(1280,1280))

    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    ret = ort_sess.run(None, { 'input_1': img } )

    print(type(img), img.shape, img.dtype, type(ret))
    for x in ret:
        print("    ", type(x), x.shape)