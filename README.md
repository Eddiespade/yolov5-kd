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
 2. 

### 调整二: 在大模型的基础上修改网络结构（如Yolov5m-CA） 



