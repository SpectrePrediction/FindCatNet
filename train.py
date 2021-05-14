import util.DateReader as dr
import numpy as np
import cv2
import models
import torchvision.transforms as transforms
import torch
import time
import os
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


@dr.data_read_function
def label_read_func(img_path, scripted_module):
    if '.' in img_path:
        return scripted_module(
                torch.from_numpy(
                    cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1).transpose((2, 0, 1)).astype("float32") / 255.0
                )
            ).numpy()

    return int(img_path)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.001
    batch_size = 8
    class_num = 13

    read_data_cache = batch_size * 2
    using_thread_num = 4
    is_completion = True
    is_show_progress = False

    train_txt_path = r"./train.txt"
    save_model_path = "./ckpt"

    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    resnext50_32x4d = models.resnext50_32x4d(pretrained=True, strict=False).to(device)

    scripted_module = torch.jit.script(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    resnext50_32x4d.fc = torch.nn.Linear(2048, class_num).to(device)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(resnext50_32x4d.parameters(), lr=learning_rate, weight_decay=5e-3)
    # print(resnext50_32x4d)

    datareader = dr.DataReader(
        train_txt_path, [],
        read_data_func=label_read_func(scripted_module),
        read_data_cache=read_data_cache,  # 读取数据的缓冲大小
        batch_size=batch_size,  # 读取数据得到的批大小 必填
        is_completion=is_completion,  # 是否填充
        using_thread_num=using_thread_num,  # 使用线程数
        is_show_progress=is_show_progress  # 是否可视化读取进程
        # read_txt_func  # 读取文本的函数 使用默认
        # is_shuffle  # 是否乱序 使用默认
    )

    scaler = GradScaler()
    temp_epoch = -1
    total_step = datareader.total // batch_size
    for epoch, image, label in datareader:
        star = time.time()
        optimizer.zero_grad()

        # image = np.array(image)# .transpose((0, 3, 1, 2)).astype("float32") / 255.0
        # image = scripted_module(torch.as_tensor(image)).to(device)
        image = torch.as_tensor(image, device=device)  # .to(device)
        label = torch.as_tensor(label, dtype=torch.long, device=device)
        # print(label.shape)
        # print(f"time{time.time()-star}")
        with autocast():
            outputs = resnext50_32x4d(image)
            loss = criterion(outputs, label)

        # print(f"epoch:{epoch}, loss:{loss.item()}, time:{time.time()-star}")

        scaler.scale(loss).backward()
        # loss.backward()
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()

        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]"
              f" epoch:{epoch}, loss:{loss}, batch_time:{time.time()-star}, "
              f"step:{datareader.ont_epoch_step - 1}/{total_step}")

        if epoch != temp_epoch:
            temp_epoch = epoch
            torch.save(resnext50_32x4d.state_dict(), os.path.join(save_model_path, f'model_{epoch}.ckpt'))
        # break


if __name__ == '__main__':
    main()