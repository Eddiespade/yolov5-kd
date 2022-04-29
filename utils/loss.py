# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """
        p: 网络输出，List[torch.tensor * 3], p[i].shape = (b, 3, h, w, nc+5), hw分别为特征图的长宽
        ,b为batch-size

        targets: targets.shape = (nt, 6) , 6=icxywh,i表示第一张图片，c为类别，然后坐标xywh为(在原图中)归一
        化后的GT框

        model: 模型

        """
        '''
        lcls,lbox,lobj分别记录分类损失，box损失，obj损失
        tcls:List(tensor*3)  tensor:(N)   为类别   值为0~nc-1
        tbox:List(tensor*3)  tensor:(N,4)   4代表grid_offx,grid_offy,gw,gh
        indices:List(tuple*3)  tuple:(b,a,竖直方向第y个格子,竖直方向第x个格子)   为特征图中格子的索引
        anch:List(tensor*3)  tensor:(N)    为anchor的索引,如0,1,2,0,0,1...  因为不同尺度用不同大小的anchor

        '''
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            '''
            b:(N) a:(N) gi:(N) gj:(N)
            这是之前target中有需要做回归的格子的索引,后续可以利用索引将网络中相应部分索引出来，
            从而使得objloss针对所有格子，而boxloss和clsloss只针对有目标的格子.
            (也就是说无对应obj的格子不进行分类和回归损失计算)
            '''
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            '''
            先设置全为0,后续根据，b,a,gi,gj这些索引，将tobj中填充正样本，其他均为负样本

            pi:(b, 3, h, w, nc+5)   最后一个维度 是按  x y h w obj + nc 这个顺序来的，所以取索引0就是取x
            ,但是我们只是要它的形状而已

            这里取索引4也是同样的结果
            tobj:(b, 3, h, w)
            '''
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


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


def feature_loss(x, y, at=True, ft=True):
    device = x[0].fea.device
    atloss = torch.zeros(1, device=device)
    ftloss = torch.zeros(1, device=device)
    for i in range(len(x)):
        if at:
            atloss += at_loss(x[i].fea, y[i].fea)
        if ft:
            ftloss += ft_loss(x[i].fea, y[i].fea)
    return atloss + ftloss, torch.cat((atloss, ftloss)).detach()


def wat_loss(x, y):
    device = x[0].fea.device
    w = torch.zeros(8, device=device)
    atloss = torch.zeros(1, device=device)
    ftloss = torch.zeros(1, device=device)
    for i in range(len(x)):
        w[i] = ft_loss(x[i].fea, y[i].fea)
    w = F.softmax(w, dim=0)

    # 计算权重
    for i in range(len(x)):
        atloss += w[i] * at_loss(x[i].fea, y[i].fea)
    return atloss + ftloss, torch.cat((atloss, ftloss)).detach()


def ft(x):
    return F.normalize(x.pow(2).mean(2).mean(2).view(x.size(0), -1))


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def ft_loss(x, y):
    # 可以修改度量函数（如余弦相似度）
    # cosin = torch.cosine_similarity(ft(x), ft(y))[0]
    return (ft(x) - ft(y)).pow(2).mean()


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


class Feature_Adap(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=1):
        super(Feature_Adap, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class CSL(nn.Module):
    def __init__(self):
        super(CSL, self).__init__()

    def forward(self, tf, sf, len, device):
        loss = []
        w_l = []
        for i in range(len):
            # 全局平均池化求得s,t的B,C 并正则化
            qs = sf[i].fea.mean(3).mean(2)
            qt = tf[i].fea.mean(3).mean(2)
            # 线性转换  s_linear : 将学生的通道维数转换到与教师保持一致  t_linear : 将教师的通道维数转换到与学生保持一致
            s_linear = nn.Linear(qs.shape[1], qt.shape[1], device=device)
            t_linear = nn.Linear(qt.shape[1], qs.shape[1], device=device)
            # 线性转换到通道数一致 并变为BC 并正则化
            qs_t = s_linear(qs.type(torch.FloatTensor).to(device))
            qt_s = t_linear(qt.type(torch.FloatTensor).to(device))

            # 分别求得教师与学生每个通道的权重
            w_t = (qt.to(device) - qs_t).pow(2).view(qt.shape[0], qt.shape[1], 1, 1)
            w_s = (qs.to(device) - qt_s).pow(2).view(qs.shape[0], qs.shape[1], 1, 1)
            # 将权重软化
            w_s_soft = F.softmax(w_s, dim=1)
            w_t_soft = F.softmax(w_t, dim=1)

            # 分别乘以各自权重
            t = torch.mul(w_t_soft, tf[i].fea)
            s = torch.mul(w_s_soft, sf[i].fea)

            # 每个C3层对应的损失
            diff = at_loss(s, t)
            loss.append(diff)

            # 层与层之间的权重
            weight = torch.add(w_t.mean(), w_s.mean()) / 2
            w_l.append(weight)

        w_l = F.softmax(torch.tensor(w_l, device=device), dim=0)

        return loss


def creat_mask_map(targets, anch_wh, imgs, feature_size):
    # 放到特征图对应的大小上
    device = imgs.device
    anch_wh = [x * feature_size / imgs.shape[3] for x in anch_wh]
    bz = (targets[targets.shape[0] - 1][0] + 1).long().item()
    mask_list = np.zeros((bz, feature_size, feature_size))
    # 用来把真实标签还原的
    gain = torch.ones(6, device=device)
    gain[2:] = torch.tensor(feature_size)
    # t为真实标签在对应特征图上的大小
    targets = targets.to(device)
    t = targets * gain
    # nt:真实标签的数量
    nt = targets.shape[0]
    for i in range(feature_size):
        for j in range(feature_size):
            iou = 0
            max_iou = 0

            for n in range(nt):
                anch_xywh1 = [i, j, anch_wh[0], anch_wh[1]]
                anch_xywh1 = torch.tensor(anch_xywh1, device=device)
                anch_xywh2 = [i, j, anch_wh[2], anch_wh[3]]
                anch_xywh2 = torch.tensor(anch_xywh2, device=device)
                anch_xywh3 = [i, j, anch_wh[4], anch_wh[5]]
                anch_xywh3 = torch.tensor(anch_xywh3, device=device)
                iou1 = bbox_iou(anch_xywh1, t[n][2:], x1y1x2y2=False)
                iou2 = bbox_iou(anch_xywh2, t[n][2:], x1y1x2y2=False)
                iou3 = bbox_iou(anch_xywh3, t[n][2:], x1y1x2y2=False)
                k = t[n][0].long().item()
                k1 = t[n - 1][0].long().item()
                if n > 0 and k == k1:
                    iou += iou1 + iou2 + iou3
                    mask_list[k][i][j] += iou
                    max_iou = max_iou if max_iou >= iou else iou
                else:
                    iou = iou1 + iou2 + iou3
                    mask_list[k][i][j] += iou
                    max_iou = max_iou if max_iou >= iou else iou

    mask_list[mask_list < max_iou.item() * 0.5] = 0
    mask_list[mask_list >= max_iou.item() * 0.5] = 1
    mask_map = torch.tensor(mask_list, device=device)

    return mask_map


def creat_mask_map_2(pred, targets, feature_size, model):
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]
    # na:anchor数量  nt:真实标签数量
    na, nt = det.na, targets.shape[0]

    tcls, tbox, indices, anch = [], [], [], []
    # 为了将gt放在feature的尺寸上面
    gain = torch.ones(7, device=targets.device)
    # ai就代表每个尺度的索引，默认里面的值为0,1,2
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    # targets： [na, nt, 7]    其中7表示: i c x y w h ai； ai就代表每个尺度的索引，默认里面的值为0,1,2
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

    # 设置偏置矩阵
    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):  # nl=>3
        # anchors 匹配需要逐层匹配
        anchors = det.anchors[i]  # shape=>[3,3,2]
        gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]
        # 将targets放在对应尺寸上面
        t = targets * gain

        if nt:
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            # 可以不过滤
            t = t[j]  # filter

            gxy = t[:, 2:4]  # 格子xy，存储以特征图==左上角==为零点的gt box的(x,y)坐标
            gxi = gain[[2, 3]] - gxy  # 取反 即：以特征图==右下角==为零点的gt box的(x,y)坐标信息
            # 这两个条件可以用来选择靠近的两个邻居网格
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]  # 过滤box
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 过滤偏置

            # b表示当前bbox属于该batch内第几张图片，c表示这张照片属于哪类
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 取整
            gi, gj = gij.T  # 网格xy位置
            # a表示当前gt box和当前层的第几个anchor匹配上了
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices

            tbox.append(torch.cat((gxy - gij, gwh), 1))  #
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class


def creat_mask_map_1(targets, imgs, feature_size):
    # 放到特征图对应的大小上
    device = imgs.device
    # 获取batch_size
    bz = (targets[targets.shape[0] - 1][0] + 1).long().item()
    # 初始化mask_list
    mask_list = np.zeros((bz, feature_size, feature_size))
    # 用来把真实标签还原的
    gain = torch.ones(6, device=device)
    gain[2:] = torch.tensor(feature_size)
    # t为5实标签在对应特征图上的大小
    targets = targets.to(device)
    t = targets * gain
    # nt:真实标签的数量
    nt = targets.shape[0]

    for n in range(nt):
        # batch_size 的id
        bi = t[n][0].long().item()
        # x1 < x2   y1 < y2  所以x1 y1  向下取整   x2 y2 向上取整
        x1 = t[n][2].item() - t[n][4].item() / 2
        x1 = math.floor(x1)
        x2 = t[n][2].item() + t[n][4].item() / 2
        x2 = math.ceil(x2)
        y1 = t[n][3].item() - t[n][5].item() / 2
        y1 = math.floor(y1)
        y2 = t[n][3].item() + t[n][5].item() / 2
        y2 = math.ceil(y2)

        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                if i < feature_size and j < feature_size:
                    mask_list[bi][i][j] = 1

    mask_map = torch.tensor(mask_list, device=device)

    return mask_map
