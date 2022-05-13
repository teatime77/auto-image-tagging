import glob
import socket
import pickle
import numpy as np
import cv2

rcv_sock = None

def getBox(scores, boxes, shape):
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
    print("    ", scores[max_score_idx].shape, box.shape, max_idx, max_score, score[max_idx], box[0, a1:a2, row, col])

    bmp_h = shape[0]
    bmp_w = shape[1]

    num_box_h = box.shape[2]
    num_box_w = box.shape[3]

    box_h = bmp_h / num_box_h
    box_w = bmp_w / num_box_w

    cy = int(row * box_h + box_h / 2) 
    cx = int(col * box_w + box_w / 2)

    return cx, cy

def readBmpFiles(img_dir):
    for img_path in glob.glob(f'{img_dir}/*.png'):
        bmp = cv2.imread(img_path)

        yield bmp

def receiveBmp():
    global rcv_sock

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    listen_sock.bind((socket.gethostname(), 1235))  # IPとポート番号を指定します
    print('bind OK')

    listen_sock.listen(5)
    print('listen OK')

    rcv_sock, address = listen_sock.accept()
    print('accept OK')

    while True:

        # full_msg = b''
        # while True:
        #     msg = rcv_sock.recv(4 * 1024 * 1024)
        #     print('rcv', len(msg), len(full_msg))
        #     if len(msg) <= 0:

        #         print('rcv end', len(msg), len(full_msg))
        #         break
        #     full_msg += msg

        print('rcv ...')
        full_msg = rcv_sock.recv(4 * 1024 * 1024)

        bmp = pickle.loads(full_msg)
        print('rcv', bmp.shape)

        yield bmp

def sendBox(cx, cy):
    obj = {
        'cx': cx,
        'cy': cy
    }

    msg = pickle.dumps(obj)

    print('send', obj)
    rcv_sock.send(msg)
    print('send OK')


def getInputImg(bmp):

    img = bmp.astype(np.float32)
    img = img / 255.0

    img = cv2.resize(img, dsize=(1280,1280))

    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]

    return img