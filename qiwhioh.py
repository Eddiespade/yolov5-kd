import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2

matplotlib.use("TkAgg")


class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out


def get_feas_by_hook(model):
    fea_hooks = []
    for i in [2, 4, 6, 9, 13, 17, 20, 23]:
    # for i in [3, 6, 9, 13, 18, 23, 27, 31]:
        m = model.model[i]
        cur_hook = HookTool()
        m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)

    return fea_hooks


if __name__ == '__main__':
    # 从测试集中读取一张图片，并显示出来
    img_path = 'datasets/VOC/images/test2007/000088.jpg'
    img = Image.open(img_path)
    imgarray = np.array(img) / 255.0

    # 将图片处理成模型可以预测的形式

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_img = transform(img).unsqueeze(0)

    model = torch.load("runs/teacher113/Yolov5m-ca/weights/best.pt")
    if model.get("model", None) is not None:
        model = model["model"]
    model.to("cpu")
    model.float()
    model.eval()

    t_f = get_feas_by_hook(model)
    teacher_pred = model(input_img)

    plt.figure(figsize=(12, 12))
    for i in range(len(t_f)):
        # mc = t_f[i].fea.mean(3, keepdim=True).mean(2, keepdim=True)
        # Mc = mc.sigmoid()
        # x = torch.mul(Mc, t_f[i].fea)
        fea = t_f[i].fea.pow(2).mean(1).permute(1, 2, 0)
        fea = fea.detach().numpy()
        # cam 绘制
        # relu操作
        # heatmap = np.maximum(fea, 0)

        # 放大到原图尺寸上
        heatmap = cv2.resize(fea, (img.size[0], img.size[1]))
        # heatmap -= np.min(heatmap)
        heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap
        # superimposed_img = heatmap * 0.5 + img
        cv2.imwrite('datasets/hot/m-ca-cam{}.jpg'.format(i), superimposed_img)

        # plt.matshow(heatmap)
        plt.subplot(1, 8, i + 1)
        # plt.imshow(fea, cmap='rainbow')
        plt.imshow(superimposed_img, cmap='rainbow')

        plt.axis('off')

    plt.show()
    plt.savefig("datasets/hot/yolov5s-heatmap.png")
