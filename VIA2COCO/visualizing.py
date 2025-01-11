import numpy as np
import os
from PIL import Image, ImageDraw
import Dataset.dataset as dataset
import matplotlib.pyplot as plt

# based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
image_ids = dataset.train_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = dataset.train_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('../balloon/train', image['file_name']))

annotations = dataset.train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = dataset.train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
  box = annotation['bbox']
  class_idx = annotation['category_id']
  x,y,w,h = tuple(box)
  draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
  draw.text((x, y), id2label[class_idx], fill='white')

plt.imshow(image)
plt.axis('off')  # 关闭坐标轴
plt.show()