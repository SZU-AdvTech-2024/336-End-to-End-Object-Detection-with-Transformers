import json
import os
import shutil
import random
import xml.etree.ElementTree as ET


def split_dataset(voc_image_dir, output_dir, train_ratio=0.8):
    # 创建 COCO 格式的目录结构
    os.makedirs(os.path.join(output_dir, "train2017"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val2017"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # 获取图片文件列表
    images = [f for f in os.listdir(voc_image_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(images)

    # 按照比例划分训练集和验证集
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    # 复制图片到目标目录
    for img in train_images:
        shutil.copy(os.path.join(voc_image_dir, img), os.path.join(output_dir, "train2017", img))
    for img in val_images:
        shutil.copy(os.path.join(voc_image_dir, img), os.path.join(output_dir, "val2017", img))

    return train_images, val_images





def build_category_set(voc_dir):
    category_set = {}
    for xml_file in os.listdir(os.path.join(voc_dir, 'Annotations')):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(voc_dir, 'Annotations', xml_file))
        root = tree.getroot()

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in category_set:
                category_set[name] = len(category_set) + 1

    return category_set

def voc_to_coco(voc_dir, image_list, save_path, category_set):
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for name, id in category_set.items():
        coco['categories'].append({
            "id": id,
            "name": name,
            "supercategory": "none"
        })

    annotation_id = 1
    image_id = 1

    for xml_file in os.listdir(os.path.join(voc_dir, 'Annotations')):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(voc_dir, 'Annotations', xml_file))
        root = tree.getroot()

        filename = root.find('filename').text
        if filename not in image_list:  # 只处理指定的图片
            continue

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        coco['images'].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            coco['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_set[name],
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            })

            annotation_id += 1

        image_id += 1

    with open(save_path, 'w') as f:
        json.dump(coco, f, indent=4)


# 为训练集和验证集生成 JSON
voc_image_dir = r"/mnt/c/Users/32907/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
output_dir = "coco2"
train_images, val_images = split_dataset(voc_image_dir, output_dir)

voc_dir = r"/mnt/c/Users/32907/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
category_set = build_category_set(voc_dir)

voc_to_coco(voc_dir, train_images, os.path.join(output_dir, "annotations", "instances_train2017.json"), category_set)
voc_to_coco(voc_dir, val_images, os.path.join(output_dir, "annotations", "instances_val2017.json"), category_set)