import numpy as np
import cv2
import models
import torchvision.transforms as transforms
import torch
import time
import os


# 用于将多gpu分布式训练模型转化成单机模型（但已修改模型保存方式，故废弃）
def main():

    model_path = r"./ckpt/model_112.ckpt"
    save_model_path = "./ckpt_dp/model_112.ckpt"

    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    model = torch.load(model_path)
    for x in list(model.keys()):
        model[x[7:]] = model.pop(x)

    print(model.keys())

    torch.save(model, save_model_path)


if __name__ == '__main__':
    main()