# YOLOV5 V6.0知识蒸馏

一些碎碎念： 3050Ti的算力太差了，yolov5s的 batch最大设为8，yolov5m的 batch最大设为4。无语子

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
# 在使用的hyp.yaml文件中添加 计算kd损失的时候会用到的参数
giou: 0.05
kd: 1.0
```
```python
    # 加载教师模型
    if opt.kd:
        print("load teacher-model from", opt.teacher_weight)
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
    matloss = torch.zeros(1, device=device)  # mean at_losses
```
```python
    # 前向传播，不更新教师网络的参数
    if opt.kd:
        with torch.no_grad():
            teacher_pred = teacher_model(imgs)
```
```python
    # 将kd损失加入到总损失，以便反向传播，更新学生网络参数
    # Forward
    with amp.autocast(enabled=cuda):
        s_f = get_feas_by_hook(model) # 获取学生网络中间层特征
        pred = model(imgs)  # forward

        if opt.kd:
            atloss = torch.zeros(1, device=device)
            with torch.no_grad():
                t_f = get_t_feas_by_hook(teacher_model) # 获取教师网络中间层特征
                teacher_pred = teacher_model(imgs)

            for i in range(len(t_f)):
                # 计算ATloss
                atloss += at_loss(s_f[i].fea, t_f[i].fea)
        
        del s_f, t_f

        loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
        # kd
        if opt.kd:
            kdloss = compute_kd_output_loss(
                pred, teacher_pred, model, opt.kd_loss_selected, opt.temperature)
        else:
            kdloss = 0
        del teacher_pred
        loss += opt.alpha * kdloss + opt.beta * atloss
        loss_items[-1] = loss

        if RANK != -1:
            loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
        if opt.quad:
            loss *= 4.
```
```python
    # 更新 mkdloss 日志
    mkdloss = (mkdloss * i + kdloss) / (i + 1)  # update mean losses
    matloss = (matloss * i + atloss) / (i + 1)  # update mean losses
    pbar.set_description(('%10s' * 2 + '%10.4g' * 7) %
                         (f'{epoch}/{epochs - 1}', mem, *mloss, mkdloss, matloss, targets.shape[0], imgs.shape[-1]))

```

 3. 新增损失函数的计算
```python
def compute_kd_output_loss(pred, teacher_pred, model, kd_loss_selected="l2", temperature=20, reg_norm=None):
    t_ft = torch.cuda.FloatTensor if teacher_pred[0].is_cuda else torch.Tensor
    t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)
    if red != "mean":
        raise NotImplementedError(
            "reduction must be mean in distillation mode!")

    KDboxLoss = nn.MSELoss(reduction="none")
    if kd_loss_selected == "l2":
        KDclsLoss = nn.MSELoss(reduction="none")
    elif kd_loss_selected == "kl":
        KDclsLoss = nn.KLDivLoss(reduction="none")
    else:
        KDclsLoss = nn.BCEWithLogitsLoss(reduction="none")
    KDobjLoss = nn.MSELoss(reduction="none")
    # per output
    for i, pi in enumerate(pred):  # layer index, layer predictions
        # t_pi  -->  torch.Size([16, 3, 80, 80, 25])
        t_pi = teacher_pred[i]
        # t_obj_scale  --> torch.Size([16, 3, 80, 80])
        t_obj_scale = t_pi[..., 4].sigmoid()
        # zero = torch.zeros_like(t_obj_scale)
        # t_obj_scale = torch.where(t_obj_scale < 0.5, zero, t_obj_scale)

        # BBox
        # repeat 是沿着原来的维度做复制  torch.Size([16, 3, 80, 80, 4])
        b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        if not reg_norm:
            t_lbox += torch.mean(KDboxLoss(pi[..., :4], t_pi[..., :4]) * b_obj_scale)
        else:
            wh_norm_scale = reg_norm[i].unsqueeze(0).unsqueeze(-2).unsqueeze(-2)
            # pxy
            t_lbox += torch.mean(KDboxLoss(pi[..., :2].sigmoid(), t_pi[..., :2].sigmoid()) * b_obj_scale)
            # pwh
            t_lbox += torch.mean(
                KDboxLoss(pi[..., 2:4].sigmoid(), t_pi[..., 2:4].sigmoid() * wh_norm_scale) * b_obj_scale)

        # Class
        if model.nc > 1:  # cls loss (only if multiple classes)
            c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, model.nc)
            if kd_loss_selected == "kl":
                kl_loss = KDclsLoss(F.log_softmax(pi[..., 5:] / temperature, dim=-1),
                                    F.softmax(t_pi[..., 5:] / temperature, dim=-1)) * (temperature * temperature)
                t_lcls += torch.mean(kl_loss * c_obj_scale)
            else:
                t_lcls += torch.mean(KDclsLoss(pi[..., 5:], t_pi[..., 5:]) * c_obj_scale)

        t_lobj += torch.mean(KDobjLoss(pi[..., 4], t_pi[..., 4]) * t_obj_scale)
    t_lbox *= h['giou'] * h['kd']
    t_lobj *= h['obj'] * h['kd']
    t_lcls *= h['cls'] * h['kd']
    bs = pred[0].shape[0]  # batch size
    mkdloss = (t_lobj + t_lbox + t_lcls) * bs
    return mkdloss


def at(x):
    # mc = x.mean(3, keepdim=True).mean(2, keepdim=True)
    # Mc = mc.sigmoid()
    # x = torch.mul(Mc, x)
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()
```
### 调整二: 在大模型的基础上修改网络结构（如Yolov5m-CA） 
 1. 添加注意力模块（添加到 models/common.py 中）
```python
# SE模块
class SELayer(nn.Module):
    def __init__(self, c1, r=16):
        super(SELayer, self).__init__()
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
# 标准卷积层 + CBAM
class ASCBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, ratio):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ASCBAM, self).__init__()
        self.ca = ChannelAttention(c1, ratio, AS=False)
        self.sa = SpatialAttention(c1, AS=False)

    def forward(self, x):
        x = self.sa(self.ca(x))
        return x



# add CBAM
class SpatialAttention(nn.Module):
    def __init__(self, in_planes, kernel_size=7, AS=False):
        super(SpatialAttention, self).__init__()
        self.conv3 = nn.Conv2d(in_planes, in_planes, 3, padding=1)
        self.conv1_1 = nn.Conv2d(in_planes, in_planes, 1)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)  # concat完channel维度为2
        self.sigmoid = nn.Sigmoid()
        self.AS = AS

    def forward(self, x):
        if self.AS:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out = self.sigmoid(self.conv1_1(self.conv3(x)))
            max_out1, _ = torch.max(max_out, dim=1, keepdim=True)
            y = torch.cat([avg_out, max_out1], dim=1)  # 沿着channel维度concat一块
        else:
            avg_out = torch.mean(x, dim=1, keepdim=True)  # 沿着channel 维度计算均值和最大值
            max_out1, _ = torch.max(x, dim=1, keepdim=True)
            y = torch.cat([avg_out, max_out1], dim=1)  # 沿着channel维度concat一块
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, AS=False):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool_3d = nn.AdaptiveAvgPool3d((in_planes, 1, 1))

        self.conv3 = nn.Conv2d(in_planes, in_planes * 2, 3)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.AS = AS

    def forward(self, x):
        if self.AS:
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool_3d(self.conv3(x)))))
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        else:
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)
```
```python
# CA模块
class CABlock(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CABlock, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out
```
 2. 修改网络结构（models/yolo.py）
```python
# 在 parse_model() 函数下照着添加
    elif m is SELayer:
        channel, re = args[0], args[1]
        channel = make_divisible(channel * gw, 8) if channel != no else channel
        args = [channel, re]
    elif m is CABlock:
        channel, re = args[0], args[1]
        channel = make_divisible(channel * gw, 8) if channel != no else channel
        args = [channel, channel, re]
    elif m is ASCBAM:
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
   [-1, 3, C3, [128]],    # 2
   [-1, 1, SELayer, [128, 4]], # 3
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, SELayer, [256, 4]], #6
   [-1, 1, Conv, [512, 3, 2]],  # 7-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, SELayer, [512, 4]], # 9
   [-1, 1, Conv, [1024, 3, 2]],  # 10 -P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SELayer, [1024, 4]], #12
   [-1, 1, SPPF, [1024, 5]],  # 13
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 9], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 17
   [ -1, 1, SELayer, [512, 4] ], #18

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 22 (P3/8-small)
   [ -1, 1, SELayer, [256, 4] ], #23

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 19], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 26 (P4/16-medium)
   [ -1, 1, SELayer, [512, 4] ], #27

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 30 (P5/32-large)
   [ -1, 1, SELayer, [1024, 4] ], #31

   [[23, 27, 31], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```



