import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch.nn as nn
import Dataset.dataset as dataset

# 自定义检测头
class CustomDetrHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.class_labels_classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.class_labels_classifier(x)

# Detr模型
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_labels):
        super().__init__()

        # 加载预训练的 DETR 模型
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,  # 设置类别数
            ignore_mismatched_sizes=True
        )

        # 替换原有的分类头
        input_dim = self.model.class_labels_classifier.in_features
        self.model.class_labels_classifier = CustomDetrHead(input_dim, num_labels)

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        # 计算损失时不使用 empty_weight
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        # 返回损失和损失字典
        loss = outputs.loss
        return loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        return dataset.train_dataloader

    def val_dataloader(self):
        return dataset.val_dataloader

# 实例化并训练模型
if __name__ == "__main__":
    # 设置训练的超参数
    lr = 1e-5
    lr_backbone = 1e-6
    weight_decay = 1e-4
    num_labels = 2  # 设置你的类别数，例如 2 类（气球和背景）

    model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay, num_labels=num_labels)

    # 配置训练器
    trainer = pl.Trainer(max_epochs=10, devices=1)  # 使用 devices 代替 gpus
    trainer.fit(model)
