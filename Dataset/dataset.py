import torchvision
import os
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
import torch



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor


    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target



def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# train_dataset = CocoDetection(img_folder='/mnt/d/document/detr_1/balloon/train', processor=processor)
# val_dataset = CocoDetection(img_folder='/mnt/d/document/detr_1/balloon/val', processor=processor, train=False)
train_dataset = CocoDetection(img_folder='coco2/train', processor=processor)
val_dataset = CocoDetection(img_folder='coco2/val', processor=processor, train=False)
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
batch = next(iter(train_dataloader))

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

#
# print(batch.keys())
# pixel_values, target = train_dataset[0]
# print(pixel_values.shape)
# print(target)


