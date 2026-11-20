# DHG-Bench 节点分类复现

本指南用于在 **Windows + Miniconda** 环境中复现 DHG-Bench 论文中的节点分类（Node Classification）实验。内容覆盖环境搭建、数据放置、单组实验运行。

复现代码来自：

```text
https://github.com/Coco-Hut/DHG-Bench
```

论文、数据集及官方实现的入口：

- 论文：Fan Li, Xiaoyang Wang, Wenjie Zhang, Ying Zhang, Xuemin Lin. *DHG-Bench: A Comprehensive Benchmark for Deep Hypergraph Learning*. ICLR 2026.
- 官方代码：`https://github.com/Coco-Hut/DHG-Bench`
- 数据集：`https://huggingface.co/datasets/Philley/DHG-Bench-dataset`

## 1. 获取代码与数据

在准备存放项目的目录中执行：

```bash
git clone https://github.com/Lzx-qwer/HGNNStudy.git
```

从 Hugging Face 下载数据集页面中的 `data.zip`。解压后，项目根目录下应当出现名为 `data` 的文件夹。

整理完成后的主要目录如下：

```text
HGNNStudy/
├── data/
│   ├── trad_data/
│   ├── hete_data/
│   ├── fair_data/
│   ├── hgcls_data/
│   └── ...
└── dhgbench/
    ├── main.py
    ├── parameter_parser.py
    ├── requirements.txt
    ├── lib_dataset/
    ├── lib_models/
    └── lib_yamls/
```

`data` 位于项目根目录，实验入口 `main.py` 位于 `dhgbench` 子目录。运行命令前先进入 `HGNNStudy\dhgbench`。

## 2. 创建 Miniconda 环境

打开 **Miniconda Prompt**，或者已在 PowerShell 中初始化 Conda 的终端。

```bash
conda create -n dhgbench -c conda-forge python=3.9.21 pip -y
conda activate dhgbench
```

创建完成后建议确认实际解释器版本：

```bash
python --version
where python
```


### 安装项目依赖

先进入实验目录，再通过 `pip` 安装 PyTorch、PyG 及相关扩展。PyG 扩展包含编译组件，直接使用普通 `pip install` 可能找不到与当前 Python/CUDA 匹配的 wheel，因此需要使用官方 wheel 索引。

CUDA 12.1 环境：

```bash
cd "D:\你的路径\HGNNStudy\dhgbench"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.2.0+cu121.html --extra-index-url https://download.pytorch.org/whl/cu121
```

本次复现涉及的主要版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.9.21 |
| PyTorch | 2.2.2 |
| torch-geometric | 2.6.1 |
| torch-cluster | 1.6.3 |
| torch-scatter | 2.1.2 |
| torch-sparse | 0.6.18 |
| torch-spline-conv | 1.2.2 |
| deeprobust | 0.2.11 |
| ipdb | 0.13.13 |
| NumPy | 1.24.3 |
| Dask | 2024.8.0 |
| pandas | 2.0.3 |

安装完成后进行快速检查：

```bash
python -c "import torch, torch_geometric; print('torch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA:', torch.cuda.is_available())"
```

如果有 NVIDIA 显卡，但 `CUDA` 显示为 `False`，先检查显卡驱动和所安装的 PyTorch 版本，不要继续运行完整实验。

## 3. 运行第一组节点分类实验

以 Cora 数据集和 HGNN 模型为例：

```bash
conda activate dhgbench
cd "D:\你的路径\HGNNStudy\dhgbench"
python main.py --dname=cora --task_type=node_cls --method=HGNN --num_seeds=5
```

参数含义：

| 参数 | 作用 |
| --- | --- |
| `--dname` | 数据集名称 |
| `--task_type=node_cls` | 固定为节点分类任务 |
| `--method` | 模型名称 |
| `--num_seeds` | 随机种子数量；设为 5 时依次运行 seed 0～4，并汇总均值与标准差 |

可以先用 `--num_seeds=1` 做一次快速运行，确认数据读取、CUDA、模型构建和评估流程都正常，再改成 `--num_seeds=5` 获取正式结果：

```bash
python main.py --dname=cora --task_type=node_cls --method=HGNN --num_seeds=1
```

## 4. 数据集合与模型名称

论文主表及补充表中使用的节点分类数据集如下：

| 用途 | `--dname` 可选值 |
| --- | --- |
| 主实验 | `cora`、`pubmed`、`coauthor_cora`、`coauthor_dblp`、`walmart-trips-100`、`trivago`、`actor`、`twitch`、`pokec`、`yelp` |
| 补充实验 | `NTU2012`、`ModelNet40` |

可替换到 `--method` 的模型代码名称如下：

```text
MLP, CEGCN, CEGAT, HGNN, HyperGCN, HCHA, LEGCN, HyperND,
PhenomNN, SheafHyperGNN, HJRL, DPHGNN, TFHNN, HNHN, UniGCNII,
AllSetformer, EDHNN, HyperGT, EHNN, TMPHN
```

不需要在命令中额外加入 `--task_type` 的别名。节点分类统一使用 `node_cls`；代码会据此进入 `lib_yamls\node_yaml` 下的对应配置。

例如更换为 Pubmed 和 HyperGCN：

```bash
python main.py --dname=pubmed --task_type=node_cls --method=HyperGCN --num_seeds=5
```

### 配置是如何找到的

一次运行会依次定位三层信息：

1. `--task_type` 决定任务配置目录，节点分类对应 `node_yaml`。
2. `--method` 决定读取哪个模型的 YAML 文件。
3. `--dname` 决定采用该文件中的哪组数据集超参数。

如果代码提示配置不存在，通常不是环境问题，而是该模型尚未提供当前数据集对应的 YAML。请先检查 `lib_yamls`，再按照相同字段结构补充配置，不要直接套用其他数据集的参数。

## 5. 必须注意的参数兼容问题

上游 README 的部分示例包含 `--is_default=True` 或 `--is_default=False`。当前代码没有把该参数声明为布尔类型，传入的字符串在条件判断中都会被当作真值，因此这两种写法都可能意外启用默认配置。

为保证实验走所选 YAML 配置，复现命令中应**完全省略** `--is_default`，让程序使用其默认值 `False`。推荐写法始终是：

```bash
python main.py --dname=数据集名 --task_type=node_cls --method=模型名 --num_seeds=5
```

## 6. 附录：论文与算法引用

### 6.1 DHG-Bench 论文

```bibtex
@inproceedings{li2026dhg,
  title={DHG-Bench: A Comprehensive Benchmark for Deep Hypergraph Learning},
  author={Li, Fan and Wang, Xiaoyang and Zhang, Wenjie and Zhang, Ying and Lin, Xuemin},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  pages={47693--47727},
  year={2026}
}
```

### 6.2 对比算法论文

| 编号 | 代码名称 | 论文 | 发表信息 |
| :---: | :--- | :--- | :--- |
| 1 | HGNN | [Hypergraph Neural Networks](https://cdn.aaai.org/ojs/4235/4235-13-7289-1-10-20190705.pdf) | AAAI 2019 |
| 2 | HyperGCN | [HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs](https://proceedings.neurips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) | NeurIPS 2019 |
| 3 | HNHN | [HNHN: Hypergraph Networks with Hyperedge Neurons](https://grlplus.github.io/papers/40.pdf) | ICML Workshop 2020 |
| 4 | HCHA | [Hypergraph Convolution and Hypergraph Attention](https://www.sciencedirect.com/science/article/abs/pii/S0031320320304404) | Pattern Recognition 2020 |
| 5 | UniGNN | [UniGNN: A Unified Framework for Graph and Hypergraph Neural Networks](https://www.ijcai.org/proceedings/2021/0353.pdf) | IJCAI 2021 |
| 6 | AllSetformer | [You Are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/forum?id=hpBTIv2uy_E) | ICLR 2022 |
| 7 | HyperND | [Nonlinear Feature Diffusion on Hypergraphs](https://proceedings.mlr.press/v162/prokopchik22a/prokopchik22a.pdf) | ICML 2022 |
| 8 | EHNN | [Equivariant Hypergraph Neural Networks](https://arxiv.org/pdf/2208.10428) | ECCV 2022 |
| 9 | LEGCN | [Semi-supervised Hypergraph Node Classification on Hypergraph Line Expansion](https://arxiv.org/pdf/2005.04843) | CIKM 2022 |
| 10 | ED-HNN | [Equivariant Hypergraph Diffusion Neural Operators](https://openreview.net/forum?id=RiTjKoscnNd) | ICLR 2023 |
| 11 | PhenomNN | [From Hypergraph Energy Functions to Hypergraph Neural Networks](https://proceedings.mlr.press/v202/wang23d/wang23d.pdf) | ICML 2023 |
| 12 | SheafHyperGNN | [Sheaf Hypergraph Networks](https://proceedings.neurips.cc/paper_files/paper/2023/file/27f243af2887d7f248f518d9b967a882-Paper-Conference.pdf) | NeurIPS 2023 |
| 13 | HJRL | [Hypergraph Joint Representation Learning for Hypervertices and Hyperedges via Cross Expansion](https://openreview.net/forum?id=fxLaL5s6UH) | AAAI 2024 |
| 14 | DPHGNN | [DPHGNN: A Dual Perspective Hypergraph Neural Networks](https://arxiv.org/pdf/2405.16616) | KDD 2024 |
| 15 | T-HyperGNNs | [T-HyperGNNs: Hypergraph Neural Networks via Tensor Representations](https://ieeexplore.ieee.org/document/10462516) | TNNLS 2024 |
| 16 | HyperGT | [Hypergraph Transformer for Semi-Supervised Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10446248) | ICASSP 2024 |
| 17 | TF-HNN | [Training-Free Message Passing for Learning on Hypergraphs](https://openreview.net/pdf?id=4AuyYxt7A2) | ICLR 2025 |


