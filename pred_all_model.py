import numpy as np
import cv2
import models
import torchvision.transforms as transforms
import torch
import os
from util.torchcam.cams import SmoothGradCAMpp
from util.torchcam.utils import overlay_mask
import util.args as args


def pred_read_func(img_path, scripted_module):
    resize_image = cv2.resize(cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1), (512, 512))
    torch_image = torch.from_numpy(resize_image.transpose((2, 0, 1)).astype("float32") / 255.0)
    out_image = scripted_module(torch_image).numpy()

    return out_image, resize_image


def get_success_rate(_out_label_list, _image_label):
    data_situation = dict()

    for _out_label in set(_out_label_list):
        count = _out_label_list.count(_out_label)
        data_situation[_out_label] = int(count)

        # print(_out_label, '出现的次数：', count)

    print(str(data_situation))

    if data_situation.__contains__(_image_label):

        true_label = data_situation[_image_label]
        total_label = len(_out_label_list)

        return true_label / total_label
    else:
        return 0.0


def pred():

    if not os.path.exists(cam_save_path) and not dont_save_cam:
        os.mkdir(cam_save_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    resnext50_32x4d = models.resnext50_32x4d(pretrained=False).to(device)

    scripted_module = torch.jit.script(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    resnext50_32x4d.fc = torch.nn.Linear(2048, class_num).to(device)

    temp_dict = dict()
    for model_path in os.listdir(model_root_path):
        model_path = os.path.join(model_root_path, model_path)
        resnext50_32x4d.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        resnext50_32x4d.eval()

        f = open(label_txt_path, 'r')
        name_dict = dict([reversed(path.split()) for path in f.readlines()])
        print(name_dict)

        acc_list = []
        for label in os.listdir(image_root_path):
            print(f"标签： {label}")

            label_path = os.path.join(image_root_path, label)
            temp_list = []
            for i, image_name in enumerate(os.listdir(label_path)):

                image_path = os.path.join(label_path, image_name)
                image, re_image = pred_read_func(image_path, scripted_module)

                re_image = cv2.cvtColor(re_image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
                cam_extractor = SmoothGradCAMpp(resnext50_32x4d, "layer4", input_shape=(3, 512, 512))
                cam_extractor._hooks_enabled = True

                resnext50_32x4d.zero_grad()

                outputs = resnext50_32x4d(torch.unsqueeze(torch.as_tensor(image, device=device), dim=0))

                activation_map = cam_extractor(outputs.squeeze(0).argmax().item(), outputs).cpu()

                result = overlay_mask(transforms.functional.to_pil_image(torch.as_tensor(re_image, device="cpu")),
                                      transforms.functional.to_pil_image(activation_map, mode='F'), alpha=0.5)

                name = name_dict.get(str(int(torch.argmax(outputs, dim=1, keepdim=True)[0][0].cpu())))
                print(f"预测:{name} in {image_name}")

                save_image = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)

                cv2.imshow("grad cam", save_image)
                cv2.waitKey(10)
                if not dont_save_cam:
                    cv2.imencode('.jpg', save_image)[1].tofile(os.path.join(cam_save_path, f"{label}{i}_预测{name}.jpg"))

                cam_extractor.clear_hooks()
                cam_extractor._hooks_enabled = False
                temp_list.append(name)

            acc = get_success_rate(temp_list, label)
            acc_list.append(acc)
            print(f" 准确率 {acc}\n")

        acc = sum(acc_list) / acc_list.__len__()
        temp_dict[model_path] = acc
        print(f"平均 {acc}")

    print(sorted(temp_dict.items(), key=lambda x: x[1]))


if __name__ == '__main__':
    class_num = 13
    model_root_path = r"./ckpt"
    label_txt_path = r'label.txt'
    image_root_path = r"original_image"
    cam_save_path = r"./pred_grad_cam"
    dont_save_cam = True
    exec(args.get_args_compile())

    pred()
