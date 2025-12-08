# 毕业论文说明文档
本仓库实现了两个基于改进Swin Transformer的无参考图像质量评估方法，分别为针对自然图像的CLEST-IQA模型和针对医学图像的AFST-MIQA模型。


## Environment(安装环境)
通过以下命令创建 conda 环境并安装所需依赖项：

```
conda create -n iqa python=3.8 
conda activate iqa
pip install -r  requirements.txt 
```

## CLEST-IQA中对比学习网络的预训练

### 快速获取对比学习特征

这里提供一个训练好的模型参数，用于快速上手，获取对比学习网络输出特征并保存到 .npy文件中：

```
python demo_feat.py --im_path sample_images/img66.bmp --model_path models/CONTRIQUE_checkpoint25.tar --feature_save_path features.npy
```

参数说明：
```
--im_path：输入图像的路径
--model_path：预训练的对比学习模型的路径
--feature_save_path：保存提取的特征的路径
```

### 对比学习预训练数据集准备

在开始预训练对比学习网络前，需要下载训练数据集：

首先创建目录mkdir training_data用于存储训练图像，具体需要下载[KADID-700k](http://database.mmsp-kn.de/kadid-10k-database.html), [AVA](https://github.com/mtobeiyf/ava_downloader), [COCO](https://cocodataset.org/#download), [CERTH-Blur](https://mklab.iti.gr/results/certh-image-blur-dataset/), [VOC](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/).

其中KADID-700k数据集包含大量的合成失真图像，具有多种失真类型和程度，将用于辅助任务的学习，其它数据集均为真实场景中的图像，用于实例判别任务的学习。

### 训练对比学习网络

首先需要下载包含图像路径和相应失真类别的 csv 文件：

```
wget -L https://utexas.box.com/shared/static/124n9sfb27chgt59o8mpxl7tomgvn2lo.csv -O csv_files/file_names_ugc.csv -q --show-progress
wget -L https://utexas.box.com/shared/static/jh5cmu63347auyza37773as5o9zxctby.csv -O csv_files/file_names_syn.csv -q --show-progress
```

### 训练主干模型
该脚本实现了跨域的对比学习网络框架，使用合成失真图像和真实失真图像联合训练网络。

```
python train_cl.py \
  --csv_file_syn path/to/synthetic_images.csv \
  --csv_file_ugc path/to/ugc_images.csv \
  --model_path ./checkpoints \
  --batch_size 256 \
  --lr 0.6 \
  --epochs 25
```

参数说明：

```
--csv_file_syn：合成失真图像的 csv 文件路径
--csv_file_ugc：真实失真图像的 csv 文件路径
--model_path：保存训练好的模型的路径
--batch_size：每个训练批次的图像数量
--lr：学习率，控制模型参数更新的步长
--epochs：训练的轮数
```


## CLEST-IQA模型的训练和测试

### 数据准备

CLEST-IQA模型所需要下载的数据集包括：四个合成失真数据集([LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm), [CSIQ](http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23), [TID2013](http://www.ponomarenko.info/tid2013.htm), [KADID10K](http://database.mmsp-kn.de/kadid-10k-database.html), 和三个真实失真数据集[LIVE challenge](https://live.ece.utexas.edu/research/ChallengeDB/), [KonIQ](http://database.mmsp-kn.de/koniq-10k-database.html), [LIVEFB](https://baidut.github.io/PaQ-2-PiQ/)).

### 训练模型

需要引入预训练对比学习网络的权重，在训练时冻结对比学习网络的参数：

```
python train_clest.py \
  --dset livec \
  --epoch 50 \
  --model_path_CONTRIQUE models/CONTRIQUE_checkpoint25.tar \
  --bsize 8
```
python train_clest.py --dset livec --epoch 50 --model_path_CONTRIQUE models/CONTRIQUE_checkpoint25.tar --bsize 8

参数说明：
```
--dset: 使用的自然图像数据集
--epoch：训练的轮数
--model_path_CONTRIQUE：预训练的对比学习网络的权重路径
--bsize：每个训练批次的图像数量
```

### 测试模型

```
python test_clest.py \
  --dset livec \
  --model_name niqa \
  --model_path_CONTRIQUE models/CONTRIQUE_checkpoint25.tar
```
python test_clest.py --dset livec --model_name niqa --model_path_CONTRIQUE models/CONTRIQUE_checkpoint25.tar

参数说明：
```
--dset: 使用的自然图像数据集
--model_name：训练好的 CLEST - IQA 模型的名称
--model_path_CONTRIQUE：对比学习网络的检查点路径
```

## AFST-MIQA模型的训练和测试

### 数据准备

AFST模型所需要下载四个医学图像质量评估数据集，分别为[PMIQD](https://github.com/mikugyf/PMIQD-SIS/tree/main/data/dataset), [FocusPath](https://drive.usercontent.google.com/download?id=1TlPszQjwhnlBU6LScKBkUa78eA7oq13X&export=download&authuser=0&confirm=t&uuid=8adc459d-5749-44e4-936d-f33bbc5836b7&at=APcmpoxyII21V_CrLOiuIdguuwQm%3A1744096443504), [MRIQA-DB](https://marosz.kia.prz.edu.pl/NOMRIQA.html), [CXIQ](https://github.com/MIRACLE-Center/CXIQ).

### 训练模型

使用冻结的 Swin Transformer 骨干网络和冻结的对比编码器训练AFST模型：

```
python train_afst.py \
  --epoch 50 \
  --dset pmiqd \
  --bsize 8 \
  --model_path_CONTRIQUE models/CONTRIQUE_checkpoint25.tar \
  --model_path_clest sav/model/niqa.pth
```
python train_afst.py --epoch 50 --dset pmiqd --bsize 8 --model_path_CONTRIQUE models/CONTRIQUE_checkpoint25.tar --model_path_clest sav/model/niqa.pth

参数说明：
```
--epoch: 训练的轮数
--dset：使用的数据集名称
--bsize：每个训练批次的图像数量
--model_path_CONTRIQUE：对比学习网络的检查点路径
--model_path_clest： 训练好的clest模型的路径
```

### 测试模型

```
python test_afst.py \
  --dset pmiqd \
  --model_path sav/model_m \
  --model_path_CONTRIQUE models/CONTRIQUE_checkpoint25.tar 
```
python test_afst.py --model_path sav/model_m --model_path_CONTRIQUE models/CONTRIQUE_checkpoint25.tar 

参数说明：
```
--dset：使用的数据集名称
--model_path：训练好的 AFST-MIQA 模型的路径
--model_path_CONTRIQUE models：对比学习网络的检查点路径
```
