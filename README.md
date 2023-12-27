# mindspore_gallery
基于MindSpore框架，复现一些经典的论文。

## GRL
### GCN
- Paper: Thomas N. Kipf, Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR 2016. [Link](https://arxiv.org/abs/1609.02907)
- Code: [Pytorch](https://github.com/tkipf/pygcn), [Tenserflow](https://github.com/tkipf/gcn)

#### Environment Requirements

- Python 3.9
- MindSpore 2.1.1
- Cuda 11.1

```bash
conda create -n ms2 python=3.9

conda activate ms2

# CUDA 11.1  MindSpore 2.1.1
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.1.1/MindSpore/unified/x86_64/mindspore-2.1.1-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# 训推
python train_grl_gcn.py --seed 42

```

