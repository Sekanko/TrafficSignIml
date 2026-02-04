import os
import shutil
import random

def prepare_yolo_dataset(source_path, target_path="detection_yolo", split_ratio=0.8):
    imgs_src = os.path.join(source_path, 'imgs')
    labels_src = os.path.join(source_path, 'labels')
    
    images = [f for f in os.listdir(imgs_src) if f.lower().endswith(('.png', '.jpg'))]
    valid_pairs = []

    for img in images:
        name = os.path.splitext(img)[0]
        label = name + ".txt"
        if os.path.exists(os.path.join(labels_src, label)):
            valid_pairs.append((img, label))

    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * split_ratio)
    
    for split, pairs in [('train', valid_pairs[:split_idx]), ('val', valid_pairs[split_idx:])]:
        img_out = os.path.join(target_path, split, 'images')
        lbl_out = os.path.join(target_path, split, 'labels')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_file, lbl_file in pairs:
            shutil.copy(os.path.join(imgs_src, img_file), os.path.join(img_out, img_file))
            shutil.copy(os.path.join(labels_src, lbl_file), os.path.join(lbl_out, lbl_file))

    return os.path.abspath(target_path)