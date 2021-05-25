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
import util.args as args


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

    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    resnext50_32x4d = models.resnext50_32x4d(pretrained=pretrained, strict=False).to(device)

    scripted_module = torch.jit.script(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    resnext50_32x4d.fc = torch.nn.Linear(2048, class_num).to(device)

    if retrain:
        state_dict = torch.load(retrain_model_path, map_location=device)
        if pretrained:
            del state_dict['fc.weight']
            del state_dict['fc.bias']
        resnext50_32x4d.load_state_dict(state_dict, strict=re_load_strict)

    # criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion = models.ContrastiveLoss().cuda()
    optimizer = torch.optim.AdamW(resnext50_32x4d.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    use_multiple_gpu = torch.cuda.device_count() > 1
    print(f"can use {torch.cuda.device_count()} GPU")
    if use_multiple_gpu:
        resnext50_32x4d = torch.nn.DataParallel(resnext50_32x4d)

    scaler = GradScaler()
    total_step = datareader.total // batch_size

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience,
                                                           cooldown=total_step * cooldown_epoch, verbose=verbose)

    for epoch, image, label in datareader:
        star = time.time()
        optimizer.zero_grad()

        image_1 = image[0:batch_size // 2]
        image_2 = image[batch_size // 2:]

        label = tuple(map(lambda x: 1 if x[0] == x[1] else 0, zip(label[0:batch_size // 2], label[batch_size // 2:])))

        # image = np.array(image)# .transpose((0, 3, 1, 2)).astype("float32") / 255.0
        # image = scripted_module(torch.as_tensor(image)).to(device)
        # image = torch.as_tensor(image, device=device)  # .to(device)
        image_1 = torch.as_tensor(image_1, device=device)
        image_2 = torch.as_tensor(image_2, device=device)
        label = torch.as_tensor(label, dtype=torch.long, device=device)
        # print(label.shape)
        # print(f"time{time.time()-star}")
        with autocast():
            # outputs = resnext50_32x4d(image)
            # loss = criterion(outputs, label)
            outputs_1 = resnext50_32x4d(image_1)
            outputs_2 = resnext50_32x4d(image_2)
            loss = criterion(outputs_1, outputs_2, label)

        # print(f"epoch:{epoch}, loss:{loss.item()}, time:{time.time()-star}")

        scaler.scale(loss).backward()
        # loss.backward()
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()

        scheduler.step(loss)

        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]"
              f" epoch:{epoch}, loss:{loss}, batch_time:{time.time()-star}, "
              f"step:{datareader.ont_epoch_step - 1}/{total_step}, lr:{optimizer.state_dict()['param_groups'][0]['lr']}")

        if epoch % save_interval == 0:

            save_state_dict = resnext50_32x4d.state_dict()
            if use_multiple_gpu:
                save_state_dict = resnext50_32x4d.module.state_dict()
            torch.save(save_state_dict, os.path.join(save_model_path, f'model_{epoch}.ckpt'))


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pretrained = True
    retrain = True
    re_load_strict = False
    retrain_model_path = r"./ckpt_final/model_347.ckpt"

    learning_rate = 0.002  # 0.0006
    batch_size = 32
    class_num = 13

    read_data_cache = batch_size * 2
    using_thread_num = 6
    is_completion = True
    is_show_progress = False
    weight_decay = 5e-3

    factor = 0.8
    patience = 10
    cooldown_epoch = 4
    verbose = False

    save_interval = 1

    train_txt_path = r"./train.txt"
    save_model_path = "./ckpt"
    exec(args.get_args_compile())

    main()
