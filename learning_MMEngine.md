# learning_MMEngine

# 15分钟上手基本训练验证流程

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



# 数据集（Dataset）与数据加载器（DataLoader）

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

细心的你可能会发现，例 1 **并非**直接由[15分钟上手](https://mmengine.readthedocs.io/zh-cn/latest/get_started/15_minutes.html)中的示例代码简单修改而来。你可能本以为将 `DataLoader` 简单替换为 `dict` 就可以无缝切换，但遗憾的是，基于注册机制构建时 MMEngine 会有一些隐式的转换和约定。我们将介绍**其中的不同点**，以避免你使用配置文件时产生不必要的疑惑。

与 15 分钟上手明显不同，例 1 中我们添加了 `sampler` 参数，这是由于**在 MMEngine 中我们要求通过 `dict` 传入的数据加载器的配置必须包含 `sampler` 参数**。同时，`shuffle` 参数也从 `DataLoader` 中移除，这是由于在 PyTorch 中 **`sampler` 与 `shuffle` 参数是互斥的**，见 [PyTorch API 文档](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。

当考虑 `sampler` 时，例 1 代码**基本**可以认为等价于下面的代码块

```python
from mmengine.dataset import DefaultSampler

dataset = torchvision.datasets.CIFAR10(...)
sampler = DefaultSampler(dataset, shuffle=True)

runner = Runner(
    train_dataloader=DataLoader(
        batch_size=32,
        sampler=sampler,
        dataset=dataset,
        collate_fn=default_collate
    )
)
```



> 上述代码的等价性只有在：1）使用单进程训练，以及 2）没有配置执行器的 `randomness` 参数时成立。这是由于使用 `dict` 传入 `sampler` 时，执行器会保证它在分布式训练环境设置完成后才被惰性构造，并接收到正确的随机种子。这两点在手动构造时需要额外工作且极易出错。因此，上述的写法只是一个示意而非推荐写法。我们**强烈建议 `sampler` 以 `dict` 的形式传入**，让执行器处理构造顺序，以避免出现问题。

### DefaultSampler

上面例子可能会让你好奇：`DefaultSampler` 是什么，为什么要使用它，是否有其他选项？事实上，`DefaultSampler` 是 MMEngine 内置的一种采样器，它屏蔽了单进程训练与多进程训练的细节差异，使得单卡与多卡训练可以无缝切换。如果你有过使用 PyTorch `DistributedDataParallel` 的经验，你一定会对其中更换数据加载器的 `sampler` 参数有所印象。但在 MMEngine 中，这一细节通过 `DefaultSampler` 而被屏蔽。

除了 `Dataset` 本身之外，`DefaultSampler` 还支持以下参数配置：

- `shuffle` 设置为 `True` 时会打乱数据集的读取顺序
- `seed` 打乱数据集所用的随机种子，通常不需要在此手动设置，会从 `Runner` 的 `randomness` 入参中读取
- `round_up` 设置为 `True` 时，与 PyTorch `DataLoader` 中设置 `drop_last=False` 行为一致。如果你在迁移 PyTorch 的项目，你可能需要注意这一点。

> 更多关于 `DefaultSampler` 的内容可以参考 [API 文档](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler)

`DefaultSampler` 适用于绝大部分情况，并且我们保证在执行器中使用它时，随机数等容易出错的细节都被正确地处理，防止你陷入多进程训练的常见陷阱。如果你想要使用基于迭代次数 (iteration-based) 的训练流程，你也许会对 [InfiniteSampler](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.InfiniteSampler.html#mmengine.dataset.InfiniteSampler) 感兴趣。如果你有更多的进阶需求，你可能会想要参考上述两个内置 `sampler` 的代码，实现一个自定义的 `sampler` 并注册到 `DATA_SAMPLERS` 根注册器中。

```python
@DATA_SAMPLERS.register_module()
class MySampler(Sampler):
    pass

runner = Runner(
    train_dataloader=dict(
        sampler=dict(type='MySampler'),
        ...
    )
)
```

### 不起眼的 collate_fn

PyTorch 的 `DataLoader` 中，`collate_fn` 这一参数常常被使用者忽略，但在 MMEngine 中你需要额外注意：当你传入 `dict` 来构造数据加载器时，MMEngine 会默认使用内置的 [pseudo_collate](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.pseudo_collate.html#mmengine.dataset.pseudo_collate)，这一点明显区别于 PyTorch 默认的 [default_collate](https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate)。因此，当你迁移 PyTorch 项目时，需要在配置文件中手动指明 `collate_fn` 以保持行为一致。

> MMEngine 中使用 `pseudo_collate` 作为默认值，主要是由于历史兼容性原因，你可以不必过于深究，只需了解并避免错误使用即可。

MMengine 中提供了 2 种内置的 `collate_fn`：

- `pseudo_collate`，**缺省时的默认参数。它不会将数据沿着 `batch` 的维度合并**。详细说明可以参考 [pseudo_collate](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.pseudo_collate.html#mmengine.dataset.pseudo_collate)
- `default_collate`，与 PyTorch 中的 `default_collate` 行为几乎完全一致，会将数据转化为 `Tensor` 并沿着 `batch` 维度合并。一些细微不同和详细说明可以参考 [default_collate](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.default_collate.html#mmengine.dataset.default_collate)

如果你想要使用自定义的 `collate_fn`，你也可以将它注册到 `FUNCTIONS` 根注册器中来使用

```python
@FUNCTIONS.register_module()
def my_collate_func(data_batch: Sequence) -> Any:
    pass

runner = Runner(
    train_dataloader=dict(
        ...
        collate_fn=dict(type='my_collate_func')
    )
)
```

## 数据集详解

数据集通常定义了**数据的数量、读取方式与预处理**，并作为参数传递给数据加载器供后者分批次加载。由于我们使用了 PyTorch 的 `DataLoader`，因此数据集也自然与 PyTorch `Dataset` 完全兼容。同时得益于注册机制，当数据加载器使用 `dict` 在执行器内部构建时，`dataset` 参数也可以使用 `dict` 传入并在内部被构建。这一点使得编写配置文件成为可能。

### 使用 torchvision 数据集

`torchvision` 中提供了丰富的公开数据集，它们都可以在 MMEngine 中直接使用，例如 [15 分钟上手](https://mmengine.readthedocs.io/zh-cn/latest/get_started/15_minutes.html)中的示例代码就使用了其中的 `Cifar10` 数据集，并且使用了 `torchvision` 中内置的数据预处理模块。

但是，当需要将上述示例转换为配置文件时，你需要对 `torchvision` 中的数据集进行额外的注册。如果你同时用到了 `torchvision` 中的数据预处理模块，那么你也需要编写额外代码来对它们进行注册和构建。下面我们将给出一个等效的例子来展示如何做到这一点。

```python
import torchvision.transforms as tvt
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose

# 注册 torchvision 的 CIFAR10 数据集
# 数据预处理也需要在此一起构建
@DATASETS.register_module(name='Cifar10', force=False)
def build_torchvision_cifar10(transform=None, **kwargs):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = Compose(transform)
    return torchvision.datasets.CIFAR10(**kwargs, transform=transform)

# 注册 torchvision 中用到的数据预处理模块
DATA_TRANSFORMS.register_module('RandomCrop', module=tvt.RandomCrop)
DATA_TRANSFORMS.register_module('RandomHorizontalFlip', module=tvt.RandomHorizontalFlip)
DATA_TRANSFORMS.register_module('ToTensor', module=tvt.ToTensor)
DATA_TRANSFORMS.register_module('Normalize', module=tvt.Normalize)

# 在 Runner 中使用
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=dict(type='Cifar10',
            root='data/cifar10',
            train=True,
            download=True,
            transform=[
                dict(type='RandomCrop', size=32, padding=4),
                dict(type='RandomHorizontalFlip'),
                dict(type='ToTensor'),
                dict(type='Normalize', **norm_cfg)])
    )
)
```



> 上述例子中大量使用了[注册机制](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/registry.html)，并且用到了 MMEngine 中的 [Compose](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.dataset.Compose.html#mmengine.dataset.Compose)。如果你急需在配置文件中使用 `torchvision` 数据集，你可以参考上述代码并略作修改。但我们更加推荐你有需要时在下游库（如 [MMDet](https://github.com/open-mmlab/mmdetection) 和 [MMPretrain](https://github.com/open-mmlab/mmpretrain) 等）中寻找对应的数据集实现，从而获得更好的使用体验。

### 自定义数据集

你可以像使用 PyTorch 一样，自由地定义自己的数据集，或将之前 PyTorch 项目中的数据集拷贝过来。如果你想要了解如何自定义数据集，可以参考 [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

### 使用 MMEngine 的数据集基类

除了直接使用 PyTorch 的 `Dataset` 来自定义数据集之外，你也可以使用 MMEngine 内置的 `BaseDataset`，参考[数据集基类](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/basedataset.html)文档。它对标注文件的格式做了一些约定，使得数据接口更加统一、多任务训练更加便捷。同时，数据集基类也可以轻松地搭配内置的[数据变换](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/data_transform.html)使用，减轻你从头搭建训练流程的工作量。

目前，`BaseDataset` 已经在 OpenMMLab 2.0 系列的下游仓库中被广泛使用。
