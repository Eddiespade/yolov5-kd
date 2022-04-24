# YOLOV5 V6.0知识蒸馏

一些碎碎念： 3050Ti的算力太差了，yolov5s的 batch最大设为8，无语子

--------------
### 调整一: 关闭了 **wandb** 
 1. `utils/loggers/wandb/wandb_utils.py`
```python
try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

# 修改为
wandb = None
```
 2. `utils/loggers/__init__.py`
```python
try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in [0, -1]:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

# 修改为
wandb = None
```
### 调整二: 在train.py的基础上增加了kd框架，名为：train_kd.py
 1. 增加默认参数的设定
```python
# 是否使用知识蒸馏
parser.add_argument('--kd', action='store_true', default=True, help='cache images for faster training')
# 预训练好的教师网络模型权重文件地址
parser.add_argument('--teacher_weight', type=str, default=ROOT / 'runs/weights/YOLOV5M-NO/weights/best.pt',
                        help='initial teacher_weight path')
# 使用何种损失函数计算蒸馏损失
parser.add_argument('--kd_loss_selected', type=str, default='l2', help='using kl/l2 loss in distillation')
# 如果使用的kl计算损失，则需要指定温度temperature
parser.add_argument('--temperature', type=int, default=20, help='temperature in distilling training')
```
 2. 在训练流程中添加kd
```python
    # 加载教师模型
    if opt.kd:
        print("load teacher-model from", opt.teacher_weight)
        # load teacher_model
        teacher_model = torch.load(opt.teacher_weight)
        if teacher_model.get("model", None) is not None:
            teacher_model = teacher_model["model"]
        teacher_model.to(device)
        teacher_model.float()
        teacher_model.train()
```
```python   
    # 用于保存 kd损失值 
    mkdloss = torch.zeros(1, device=device)  # mean kd_losses
```
```python
    # 前向传播，不更新教师网络的参数
    if opt.kd:
        with torch.no_grad():
            teacher_pred = teacher_model(imgs)
```
```python
    # 将kd损失加入到总损失，以便反向传播，更新学生网络参数
    if opt.kd:
        kdloss = compute_kd_output_loss(
            pred, teacher_pred, model, opt.kd_loss_selected, opt.temperature)
    else:
        kdloss = 0
    loss += kdloss
    loss_items[-1] = loss
```
```python
    # 更新 mkdloss 日志
    mkdloss = (mkdloss * i + kdloss) / (i + 1)  # update mean losses
    pbar.set_description(('%10s' * 2 + '%10.4g' * 6) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, mkdloss, targets.shape[0], imgs.shape[-1]))

```
### 调整二: 在大模型的基础上修改网络结构（如Yolov5m-CA） 
 1. 添加注意力模块（添加到 models/common.py 中）
```python
# SE模块
class SE(nn.Module):
    def __init__(self, c1, r=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
```
```python
# CBAM模块

```
```python
# CA模块

```
 2. 修改网络结构（models/yolo.py）
```python
# 在 parse_model() 函数下照着添加
    elif m is SE:
        channel, re = args[0], args[1]
        channel = make_divisible(channel * gw, 8) if channel != no else channel
        args = [channel, re]
```
 3. 修改配置文件--在每个C3模块后面添加ATM（如修改 models/yolov5m.yaml）
```yaml
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 20  # number of classes
depth_multiple: 0.67  # model depth multiple
width_multiple: 0.75  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```



