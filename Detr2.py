import pytorch_lightning as pl
from torch import nn
from transformers import DetrForObjectDetection
import torch
import Dataset.dataset as dataset
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import ModelCheckpoint
import signal
import sys

# 设置 float32 矩阵乘法精度以利用 Tensor Cores
torch.set_float32_matmul_precision('high')
train_dataloader = dataset.train_dataloader
val_dataloader = dataset.val_dataloader
batch = dataset.batch
id2label = dataset.id2label

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
        for k, v in loss_dict.items():
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
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "validation_loss"
        }

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
# 设置检查点回调
checkpoint_callback = ModelCheckpoint(
    monitor='validation_loss',
    dirpath='./checkpoints/',
    filename='detr-{epoch:02d}-{validation_loss:.2f}',
    save_top_k=3,
    mode='min'
)
# 定义一个函数来处理中断信号
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    trainer.should_stop = True  # 告诉trainer停止训练

    # 尝试保存模型
    try:
        print("Attempting to save the latest checkpoint before exiting.")
        trainer.save_checkpoint("./checkpoints/latest_checkpoint.ckpt")
        print("Latest checkpoint saved.")

        # 上传模型和处理器到 Hugging Face Hub
        print("Uploading the model and processor to Hugging Face Hub.")
        model.model.push_to_hub("hzli/baDetr")
        dataset.processor.push_to_hub("hzli/baDetr")
        print("Upload completed.")
    except Exception as e:
        print(f"Failed to save or upload: {e}")

    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
#
# outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
# print(outputs.logits.shape)

# 使用混合精度训练
trainer = Trainer(max_steps=400, gradient_clip_val=0.1, precision=16)
try:
    # 开始训练
    trainer.fit(model)
except KeyboardInterrupt:
    print("Training interrupted by user.")
# trainer.fit(model)

# model.model.push_to_hub("hzli/baDetr")
# dataset.processor.push_to_hub("hzli/baDetr")
# 训练结束后尝试上传模型
try:
    print("Training complete. Uploading the final model and processor to Hugging Face Hub.")
    model.model.push_to_hub("hzli/baDetr")
    dataset.processor.push_to_hub("hzli/baDetr")
    print("Upload completed.")
except Exception as e:
    print(f"Failed to upload: {e}")