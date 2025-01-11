一、使用原始detr复现

点击查看下载[detr论文源码](https://github.com/facebookresearch/detr)，然后修改相关的路径

点击查看下载[coco数据集](https://cocodataset.org/)，修改对应数据格式如下

```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

安装相关的环境包

从0开始训练，运行下载的detr源码中的main.py文件，运行的相关命令如下

```
python  main.py --coco_path /detr/path/ --epochs 50 --lr 0.125e-4 --lr_backbone 0.125e-5
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path /coco/ --epochs 50 --lr 0.125e-4 --lr_backbone 0.125e-5 --batch_size 16 --output_dir result-model
```

预训练，下载模型文件放在premodel文件夹下，运行下面命令

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --coco_path /coco/ --epochs 20 --lr 0.125e-4 --lr_backbone 0.125e-5 --batch_size 8 \
    --resume /premodel/detr-r50-e632da11.pth --output_dir result-model
```

更换下载数据集为[voc数据集](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)，使用提交源码中voctococo.py将voc数据集格式转成coco数据集格式，再执行下面的命令

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --coco_path /coco2 --epochs 20 --lr 0.125e-4 --lr_backbone 0.125e-5 --batch_size 8 \
    --resume /premodel/detr-r50-e632da11.pth --output_dir result-model
```

二、简洁代码实现参照了[detr源码](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb)，而提交的源码完善了优化了代码结构和引入了多项优化措施，如使用 `StepLR` 学习率调度器、启用混合精度训练，增加了提示词微调

点击下载[balloon数据集](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)，运行VIA2COCO文件夹下的to.py文件转换成代码使用的数据集格式

```
path/to/ballon/
  train/    # train images and custom_train.json
  val/      # val images and custom_val.json
```

修改的源码中的文件说明如下

```
eval.py  #评估模型
Detr.py  #模型训练
Detr2.py #对Detr.py基础上进行了相关优化
detr_interrupt_controller.py #添加中断控制
promptTuning.py  #添加了提示词微调，但是效果远达不到论文效果,有待优化
Dataset/dataset.py #数据加载
inference.py  #实现目标检测
```

