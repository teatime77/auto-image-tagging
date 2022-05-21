import os
import math
import numpy as np
import json
import cv2
import shutil

class ODTK:
    def __init__(self, output_dir, image_classes):
        self.output_dir = output_dir

        img_dir = f'{output_dir}/img'
        if os.path.isdir(img_dir):
            # フォルダがすでにある場合

            # フォルダを削除する。
            shutil.rmtree(img_dir)

        os.makedirs(img_dir, exist_ok=True)

        self.AnnoObj = {
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

            self.AnnoObj["categories"].append(o)

    def images_cnt(self):
        return len(self.AnnoObj["images"])

    def add_image(self, class_idx, video_idx, pos, compo_img, corners2, bounding_box):
        height, width = compo_img.shape[:2]

        image_id = len(self.AnnoObj["images"]) + 1

        file_name = f'{class_idx}-{video_idx}-{pos}-{image_id}.jpg'

        img_path = f'{self.output_dir}/img/{file_name}'
        cv2.imwrite(img_path, compo_img)

        self.AnnoObj["images"].append({
            "id" : image_id,
            "width": width,
            "height": height,
            "file_name" : file_name            
        })

        anno_id = len(self.AnnoObj["annotations"]) + 1
        self.AnnoObj["annotations"].append({
            "id" : anno_id,
            "image_id" : image_id, 
            "category_id" : class_idx + 1,
            "bbox" : bounding_box ,
            "segmentation" : corners2,
            "area": bounding_box[2] * bounding_box[3],           # w * h. Required for validation scores
            "iscrowd": 0            # Required for validation scores            
        })


    def save(self):
        with open(f'{self.output_dir}/train.json', 'w') as f:
            json.dump(self.AnnoObj, f, indent=4)



# how to define theta
# https://github.com/NVIDIA/retinanet-examples/issues/183#issuecomment-617860660
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
