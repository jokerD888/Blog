# learning_MMEngine

## 基本训练验证流程

下面主要是为了梳理大致流程，其中的某些细节暂时忽略。

### 构建模型

在 MMEngine 中，我们约定这个模型应当继承 `BaseModel`，并且其 `forward` 方法除了接受来自数据集的若干参数外，还需要接受**额外的参数 `mode`**：对于训练，我们需要 `mode` 接受字符串 “loss”，并返回一个包含 “loss” 字段的字典；对于验证，我们需要 `mode` 接受字符串 “predict”，并返回同时包含预测信息和真实信息的结果。

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

### 构建数据集和数据加载器

对于基础的训练和验证功能，我们可以直接使用符合 PyTorch 标准的数据加载器和数据集。

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))
```

### 构建评测指标

为了进行验证和测试，我们需要定义模型推理结果的**评测指标**。我们**约定这一评测指标需要继承 `BaseMetric`，并实现 `process` 和 `compute_metrics` 方法**。其中 `process` 方法接受数据集的输出和模型 `mode="predict"` 时的输出，此时的数据为一个批次的数据，对这一批次的数据进行处理后，保存信息至 `self.results` 属性。 而 `compute_metrics` 接受 `results` 参数，这一参数的输入为 `process` 中保存的所有信息 （如果是分布式环境，`results` 中为已收集的，包括各个进程 `process` 保存信息的结果），利用这些信息计算并返回保存有评测指标结果的字典。

```python
from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(accuracy=100 * total_correct / total_size)
```

### 构建执行器并执行任务

最后，我们利用构建好的**模型**，**数据加载器**，**评测指标**构建一个**执行器 (Runner)**，同时在其中配置 **优化器**、**工作路径**、**训练与验证配置**等选项，即可通过调用 `train()` 接口启动训练：

```python
from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    # 用以训练和验证的模型，需要满足特定的接口需求
    model=MMResNet50(),
    # 工作路径，用以保存训练日志、权重文件信息
    work_dir='./work_dir',
    # 训练数据加载器，需要满足 PyTorch 数据加载器协议
    train_dataloader=train_dataloader,
    # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # 训练配置，用于指定训练周期、验证间隔等信息
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    # 验证数据加载器，需要满足 PyTorch 数据加载器协议
    val_dataloader=val_dataloader,
    # 验证配置，用于指定验证所需要的额外参数
    val_cfg=dict(),
    # 用于验证的评测器，这里使用默认评测器，并评测指标
    val_evaluator=dict(type=Accuracy),
)

runner.train()
```

简单来说上面的重点有：

- 模型应当继承 `BaseModel`，且 `forward` 方法有额外的参数 `mode`
- 评测指标需要继承 `BaseMetric`，并实现 `process` 和 `compute_metrics` 方法
- 最后通过构建执行器，像搭积木一样，把各个模块组装起来。

## 数据集（Dataset）与数据加载器（DataLoader）

接下来逐步扩展上面流程中的各个部分。

通常来说，数据集定义了数据的总体数量、读取方式以及预处理，而数据加载器则在不同的设置下迭代地加载数据，如批次大小（`batch_size`）、随机乱序（`shuffle`）、并行（`num_workers`）等。数据集经过数据加载器封装后构成了数据源。接下来，我们将按照从外（数据加载器）到内（数据集）的顺序，逐步介绍它们在 MMEngine 执行器中的用法，并给出一些常用示例。

## 数据加载器详解

在执行器（`Runner`）中，你可以分别配置以下 3 个参数来指定对应的数据加载器

- `train_dataloader`：在 `Runner.train()` 中被使用，为模型提供训练数据
- `val_dataloader`：在 `Runner.val()` 中被使用，也会在 `Runner.train()` 中每间隔一段时间被使用，用于模型的验证评测
- `test_dataloader`：在 `Runner.test()` 中被使用，用于模型的测试

MMEngine 完全支持 PyTorch 的原生 `DataLoader`，因此上述 3 个参数均可以直接传入构建好的 `DataLoader`，比如上一章。同时，借助 MMEngine 的[注册机制](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/registry.html)，以上参数也可以传入 `dict`，如下面代码（以下简称例 1）所示。字典中的键值与 `DataLoader` 的构造参数一一对应。

```python
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=torchvision.datasets.CIFAR10(...),
        collate_fn=dict(type='default_collate')
    )
)
```

🎗️在这种情况下，数据加载器会在实际被用到时，在执行器内部被构建。

