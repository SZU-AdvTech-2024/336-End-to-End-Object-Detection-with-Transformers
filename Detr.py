import pytorch_lightning as pl
from torch import nn
from transformers import DetrForObjectDetection
import torch
import Dataset.dataset as dataset
from pytorch_lightning import Trainer
# 设置 float32 矩阵乘法精度以利用 Tensor Cores
torch.set_float32_matmul_precision('high')
train_dataloader = dataset.train_dataloader
val_dataloader = dataset.val_dataloader
batch = dataset.batch
id2label = dataset.id2label
print(id2label)
print(len(id2label))

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
         pixel_values = batch["pixel_values"]
         pixel_mask = batch["pixel_mask"]
         labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

         outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

         loss = outputs.loss
         loss_dict = outputs.loss_dict

         return loss, loss_dict

     def training_step(self, batch, batch_idx):
         loss, loss_dict = self.common_step(batch, batch_idx)
         # logs metrics for each training_step,
         # and the average across the epoch
         self.log("training_loss", loss)
         for k, v in loss_dict.items():
             self.log("train_" + k, v.item())

         return loss
     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
print(outputs.logits.shape)
# trainer = Trainer(max_steps=400, gradient_clip_val=0.1)
trainer = Trainer(max_steps=400, gradient_clip_val=0.1)

trainer.fit(model)

model.model.push_to_hub("hzli/baDetr")
dataset.processor.push_to_hub("hzli/baDetr")