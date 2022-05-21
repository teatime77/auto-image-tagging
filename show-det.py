import sys
import shutil
import numpy as np
import time
import json
import cv2

if __name__ == '__main__':
    img_dir = sys.argv[1]
    detections_json = sys.argv[2]

    cv2.namedWindow('window')

    while True:
        with open(detections_json) as f:
            df = json.load(f)

        # for cat in df['categories']:
        #     print(cat['id'], cat['name'])

        for img_inf in df['images']:
            anns_id = [ x for x in df['annotations'] if x['image_id'] == img_inf['id'] ]
            max_score = max([ x['score'] for x in anns_id ])
            anns = [ x for x in anns_id if 0.8 < x['score'] ]
            # if len(anns) == 0:
            anns = [ x for x in anns_id if x['score'] == max_score ]
            print('%s %d' % (img_inf['file_name'], 100 * max_score))
            # continue


            file_name = img_inf['file_name']
            img_path = f'{img_dir}/{file_name}'
            img = cv2.imread(img_path)

            for ann in anns:
                if len(ann['bbox']) == 4:
                    x, y, w, h = [ int(i) for i in ann['bbox']]
                    img = cv2.rectangle(img,(x,y),(x + w,y + h),(0,255,0),3)

                else:
                    x, y, w, h, theta = [ float(i) for i in ann['bbox']]

                    cx = x + 0.5 * w
                    cy = y + 0.5 * h

                    minx, miny = [ cx - 0.5 * w, cy - 0.5 * h ]
                    corners = np.array([ [ minx, miny ], [ minx + w, miny ], [ minx + w, miny + h ], [ minx, miny + h ]  ])
                    centre = np.array([cx, cy])

                    # cv2.rectangle(img, np.int0(corners[0,:]), np.int0(corners[2,:]), (0,255,0),3)

                    theta = - theta
                    rotation = np.array([ [ np.cos(theta), -np.sin(theta) ],
                                        [ np.sin(theta),  np.cos(theta) ] ])

                    corners = np.matmul(corners - centre, rotation) + centre
                    corners = np.int0(corners)
                    cv2.drawContours(img,[ corners ], 0, (255,0,0),2)




                    # v = [ int(x) for x in ann['segmentation'][0] ]
                    # print(v)
                    # img = cv2.rectangle(img,(v[0],v[1]),(v[4],v[5]),(0,255,0),3)


            cv2.imshow('window', img)
            # cv2.waitKey(0)

            k = cv2.waitKey(0)
            if k == ord('q'):
                break
            # time.sleep(1)
            # print(len(anns))


    cv2.destroyAllWindows()