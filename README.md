[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/8oH8aWc3)

## 2024年春OUC计算机视觉 期末大作业 项目说明

- 请点击此处访问亓林老师的[计算机视觉课程](https://github.com/qilin512/OUC-ComputerVision)😍

### 方案选择

- **这是一个论文复现项目**
- 我们选择基于[《基于双重注意力网络的高动态范围图像重建》](https://kns.cnki.net/kcms2/article/abstract?v=f1ZyUc11mdp2Qm0cZuNbrjJiBOJ7oHoKX0mQCajH5KW61RJgv1UjTeS75D9cV5CYQRGjypth9MSb487U0hLVOBefSFJLv-TqOJ_DS2rBz-hTC6EI-d2Wf_O7zistXOA25XuJg81ef3Y=&uniplatform=NZKPT&language=CHS)和[《Visual-Salience-Based Tone Mapping for High Dynamic Range Images》](https://ieeexplore.ieee.org/abstract/document/6779648)两篇文章进行论文**复现**，并对前文已有方法进行了改进。将现有方法进行复现和融合，得到了**一种基于双重注意力网络的高动态范围图像重建及色调映射方法**👀复现参考的论文已传至该仓库reference paper文件夹中

### 复现演示地址

- 请查看我们的bilibili视频
- 欢迎为我们的[b站视频](https://www.bilibili.com/video/BV1rpgaeCE55/?vd_source=9e77deab9cbf476a360f590847f021a1)一键三连🤩🥳

### 项目背景

- 当前市场上的摄影设备所捕捉到的图像动态范围往往**远不及自然环境**。这种低动态范围LDR的图像常常会出现过度曝光或曝光不足的区域，导致图像细节的丢失和色彩饱和度的下降。与此相对，高动态范围HDR技术能够提供更宽广的亮度范围、更丰富的色彩表现以及更完整的细节展示，因此受到广泛关注
- 目前大多数传统显示设备只有有限的动态范围，**LDR设备无法显示HDR图像**。HDR图像与显示设备的范围存在巨大差异，因此有必要对HDR图像进行压缩，以便在这些普通的LDR显示设备上同时再现极端的光影区域，使得图片在LDR设备上的显示效果更为接近实际HDR图像的真实观感

### 方法简介

- 在高动态范围(High Dynamic Range，HDR)图像重建任务中，当输入图像过曝光或者欠曝光时，常见的基于深度学习的 HDR 图像重建方法容易出现细节信息丢失和色彩饱和度差的问题。为了解决这一问题，本文提出了一种基于双重注意力网络的高动态范围图像重建及色调映射方法。

- 首先，该方法利用**双重注意力模块(Dual Attention Module，DAM)**分别从像素和通道两个维度的注意力机制对过曝光和欠曝光的两张源图像进行特征提取并融合，得到一张初步融合图像。接着，在此基础上构建**特征增强模块(Feature Enhancement Module，FEM)**分别对初步融合图像进行细节增强和颜色校正。然后，将图像传入**色调映射模块(Visual-Salience-Based Tone Mapping，VsbTM)**，利用显著性感知加权和所提出的滤波器的HDR图像局部色调映射算法，改善注意显著区域的视觉质量。最后，引用对比学习使生成图像更加接近参考图像的同时远离源图像。

  <img src="https://p.ipic.vip/xfcxgz.jpg" alt="image-20240620232458720" style="zoom: 67%;" />

- 经过多次训练，最终生成 HDR 图像。实验结果表明本文方法在 PSNR、SSIM和 LPIPS 指标上取得最优评价结果，生成的 HDR 图像色彩饱和度好且细节信息精准完整，且HDR 图像色调映射LDR图像无色晕伪影。

### 复现文档

- 技术文档已传至该仓库，为**论文复现-基于双重注意力网络的高动态范围图像重建及色调映射.pdf**📄

### 功能说明

- 传入一张.jpg或者.png格式的照片🌄，通过双重注意力模块DAM，特征增强模块FEM后，便可以得到一张修复过的.png文件，并评判它的各项指标，之后将该图片转为.hdr照片🌅，将.hdr照片传入色调映射模块VsbTM，即可得到符合个人用户显示屏的无色晕伪影的可用显示屏观看的LDR照片🌠

### 技术选型

- Python 3.9

- Matlab 2020

  ```python
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torchvision.transforms as transforms
  from torchvision.models import vgg16
  from PIL import Image
  import matplotlib.pyplot as plt
  from skimage.metrics import structural_similarity as ssim
  from lpips import LPIPS
  ```

| 技术             | 版本   |            说明            |
| ---------------- | ------ | :------------------------: |
| **PyTorch**      | 1.12.1 |      开源的机器学习库      |
| **Torchvision**  | 0.18   |      PyTorch的扩展包       |
| **Pillow**       | 10.3.0 |         图像处理库         |
| **Matplotlib**   | 3.9.0  |           绘图库           |
| **scikit-image** | 0.24.0 |  用于图像处理的Python库-   |
| **LPIPS**        | 0.1.4  | 用于计算图像感知相似度的库 |

### 数据集

- 本次实验采用公开的 [SICE数据集](https://github.com/csjcai/SICE)。SICE数据集包含一系列曝光程度的图像，其中大部分都是含有6张由高到低曝光程度图片和一张参考图片，我们选取了多组数据作为我们的实验数据。并提取数据集文件夹中低曝光图，高曝光图和参考图像作为输入和结果参考。

### 成果展示

- 双重注意力模块DAM及特征增强模块FEM

  <img src="https://p.ipic.vip/hmfe8g.png" alt="merged_image" style="zoom:33%;" />

  

  <img src="https://p.ipic.vip/l5iogz.png" alt="image-20240623103119019" style="zoom: 67%;" />

  <img src="https://p.ipic.vip/2p0s91.png" alt="merged_image (1)" style="zoom:33%;" />

  <img src="https://p.ipic.vip/ecl2q0.jpg" alt="image-20240623103204946" style="zoom:67%;" />

  <img src="https://p.ipic.vip/5oniad.jpg" alt="merged_image (2)" style="zoom:33%;" />

  <img src="https://p.ipic.vip/zzw3uy.jpg" alt="image-20240623103233674" style="zoom: 67%;" />

  <img src="https://p.ipic.vip/zy1g5f.png" alt="merged_image (3)" style="zoom:33%;" />

  <img src="https://p.ipic.vip/kn1476.jpg" alt="image-20240623103306018" style="zoom:67%;" />

  

- 色调映射模块VsbTM工作

  <img src="https://p.ipic.vip/5sgta2.png" alt="figb1" style="zoom:33%;" /><img src="https://p.ipic.vip/9qk01i.jpg" alt="figb2" style="zoom:33%;" /><img src="https://p.ipic.vip/80x2e7.png" alt="figb3" style="zoom:33%;" /><img src="https://p.ipic.vip/v5upha.png" alt="figb4" style="zoom:33%;" /><img src="https://p.ipic.vip/zml5g7.jpg" alt="figb5" style="zoom:33%;" />

  <img src="https://p.ipic.vip/uy9v2s.png" alt="figb6" style="zoom:33%;" /><img src="https://p.ipic.vip/pbqtdt.png" alt="figb7" style="zoom:33%;" /><img src="https://p.ipic.vip/1092cm.jpg" alt="figb8" style="zoom:33%;" /><img src="https://p.ipic.vip/shv0ri.jpg" alt="figb9" style="zoom:33%;" /><img src="https://p.ipic.vip/7e60x3.png" alt="figb10" style="zoom:33%;" />

  <img src="https://p.ipic.vip/lquyt4.png" alt="fig1" style="zoom:33%;" /><img src="https://p.ipic.vip/a9fcve.png" alt="fig2" style="zoom:33%;" /><img src="https://p.ipic.vip/7zns6c.png" alt="fig3" style="zoom:33%;" /><img src="https://p.ipic.vip/dff8e6.png" alt="fig4" style="zoom:33%;" /><img src="https://p.ipic.vip/vg9zns.png" alt="fig5" style="zoom:33%;" />

  <img src="https://p.ipic.vip/j8pebz.png" alt="fig6" style="zoom:33%;" /><img src="https://p.ipic.vip/bqabh4.png" alt="fig7" style="zoom:33%;" /><img src="https://p.ipic.vip/u5r7xd.png" alt="fig8" style="zoom:33%;" /><img src="https://p.ipic.vip/apb1sa.png" alt="fig9" style="zoom:33%;" /><img src="https://p.ipic.vip/r8dwcq.png" alt="fig10" style="zoom:33%;" />

- 原始的HDR图片

  <img src="https://p.ipic.vip/wkb43l.png" alt="figb1" style="zoom: 67%;" />

  <img src="https://p.ipic.vip/hnhvpm.png" alt="fig1" style="zoom:50%;" />

  

  

  

- 色调映射后的LDR图片

  <img src="https://p.ipic.vip/60gzxo.png" alt="figb10" style="zoom: 67%;" />
  
  <img src="https://p.ipic.vip/j9nkkx.png" alt="fig10" style="zoom:50%;" />

### 小组分工

- **[陈子豪](https://github.com/Chenzihao37)（45%**）：主要负责第一篇论文的复现，构建双重注意力模块DAM，特征增强模块FEM及损失函数，并进行指标分析和相关内容介绍

- **[胡楷](https://github.com/wushirenhk?tab=repositories)（42%）**：主要负责模型色调映射模块VsbTM的复现工作，视频剪辑，技术文档撰写及项目维护
- **[阮明航](https://github.com/shiper-rmh)（13%）**：技术文档审稿修改

### 致谢

- 我们的复现文章受文章[《基于双重注意力网络的高动态范围图像重建》](https://kns.cnki.net/kcms2/article/abstract?v=f1ZyUc11mdp2Qm0cZuNbrjJiBOJ7oHoKX0mQCajH5KW61RJgv1UjTeS75D9cV5CYQRGjypth9MSb487U0hLVOBefSFJLv-TqOJ_DS2rBz-hTC6EI-d2Wf_O7zistXOA25XuJg81ef3Y=&uniplatform=NZKPT&language=CHS)和文章[《Visual-Salience-Based Tone Mapping for High Dynamic Range Images》](https://ieeexplore.ieee.org/abstract/document/6779648)启发，再次感谢🥰。
- 感谢团队成员胡楷、陈子豪、阮明航的付出。





​																				 							胡楷 2024.06.23
