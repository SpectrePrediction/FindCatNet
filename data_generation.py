from xml.etree import ElementTree as Et
import cv2
import numpy as np
import random
import os
'''
    定义read_xml_info读取数据格式
    {
        label: xxx, # 取决于folder
        filename: xxx,  # 取决于filename
        class:[
            (xmin,ymin,xmax,ymax),
            (xmin,ymin,xmax,ymax),
            ...
        ]
    }
    '''


def read_xml_info(xml_path)->dict:
    """
    读取单个xml info并返回约定格式数据
    :param xml_path: xml地址
    :return: 定义格式数据
    """
    temp_dict = dict()
    tree = Et.parse(xml_path)
    root = tree.getroot()

    temp_dict['label'] = root.find('folder').text
    temp_dict['filename'] = root.find('filename').text

    for object_info in root.iter('object'):
        name = object_info.find('name').text
        bndbox_info = object_info.find('bndbox')
        bndbox_tuple = (
            int(bndbox_info.find('xmin').text),
            int(bndbox_info.find('ymin').text),
            int(bndbox_info.find('xmax').text),
            int(bndbox_info.find('ymax').text)
        )

        if temp_dict.get(name, None) is None:
            temp_dict[name] = [bndbox_tuple, ]
        else:
            temp_dict[name].append(bndbox_tuple)

    return temp_dict


def read_xml_by_root(root_path)->list:
    """
    从跟目录中读取全部xml数据并返回列表
    列表其中存放全部定义格式数据
    :param root_path: xml跟目录（即存放全部xml的目录）
    :return: list -> 存放read_xml_info的列表
    """
    temp_list = list()
    for path in os.listdir(root_path):
        if path.endswith('.xml'):

            temp_list.append(
                read_xml_info(
                    os.path.join(root_path, path)
                )
            )

    return temp_list


def split_roi_and_background(image_root_path, xml_info_list, black_list):
    """
    根据黑名单，分割兴趣区域和背景图
        1.其中背景图中失去的部分（包括黑名单类别）将会被填充修复
            其修复逻辑为左右填补，用左右的图像中较大的部分进行复制和裁剪填补
            请注意尽量不要出现兴趣区域占据整个图像的weight，否则填充难以继续
            图像填充后会进行模糊，优先对填补部分模糊后再对相交部分额外20像素进行中值滤波
        2.黑名单中的类别不会被添加到类别图像列表，但是图像上相关地方会被填充修复

        e.g:例如存在一张图 其中因为有干扰的猫 为其标注为“other cat”
            其余正常猫标注任意皆可，这里举例标注“cat”（可多标记，不影响，此处从简同一给与类别cat）
            进行处理时设置黑名单[“other cat”]，程序会将”cat“类和“other cat”裁剪出来
            其中”cat“类根据info信息中类别信息添加到列表会在最后处理完后返回
            “other cat”类别裁剪但随后被丢弃
            图像对被裁剪区域进行修补填充，将填充后的背景图放到背景图列表并在处理完后返回

    :param image_root_path: 图像根目录
    :param xml_info_list: 通过读取的带有预定义格式的列表
    :param black_list: 黑名单列表，其中类别的图像会填补但不保留
    :return: roi_图像dict(其类别是key，类别取决与info中的folder）、背景_图像list
    """
    # temp_image_list = list()
    temp_image_dict = dict()
    temp_background_list = list()
    for info in xml_info_list:
        image = cv2.imdecode(
            np.fromfile(
                os.path.join(image_root_path, info['label'], info['filename']),
                dtype=np.uint8
            ),
            -1
        )
        # shape_info = image.shape

        for key, values in tuple(info.items())[2:]:
            for x_min, y_min, x_max, y_max in values:

                if key not in black_list:
                    if temp_image_dict.get(info['label'], None):
                        temp_image_dict[info['label']].append(
                            image[y_min:y_max, x_min:x_max].copy()
                        )
                    else:
                        temp_image_dict[info['label']] = [
                            image[y_min:y_max, x_min:x_max].copy()
                        ]

                crop_flag = x_min > (image.shape[1] - x_max)
                crop_image_len = x_min if x_min > 0 and crop_flag else (image.shape[1] - x_max)
                i_len = int((x_max - x_min) / crop_image_len) + 1

                temp_image = image[y_min:y_max, :x_min] if x_min > 0 and crop_flag else image[y_min:y_max, x_max:]
                crop_image = np.hstack(
                    [temp_image for i in range(i_len)]
                )

                image[y_min:y_max, x_min:x_max] = cv2.blur(crop_image[:, :x_max - x_min], (5, 5))
                # print(info['filename'])
                # print(y_min, y_max, x_min, x_max)

                # y_min, y_max, x_min, x_max = y_min - 20, y_max + 20, x_min - 20, x_max + 20
                narray = np.array((y_min - 20, y_max + 20, x_min - 20, x_max + 20), dtype=np.int32)
                narray = np.clip(narray, 0, max(image.shape))
                y_min, y_max, x_min, x_max = tuple(map(int, tuple(narray)))

                # y_min = y_min if y_min >= 0 else 0
                # y_max = y_max if y_max >= 0 else 0
                # x_min = x_min if x_min >= 0 else 0
                # x_max = x_max if x_max >= 0 else 0
                # print(y_min, y_max, x_min, x_max)
                image[y_min:y_max, x_min:x_max] = cv2.blur(
                    image[y_min:y_max, x_min:x_max], (5, 5)
                )

        temp_background_list.append(image)

    return temp_image_dict, temp_background_list


def print_image_shape_mean(image_dict):
    """
    根据信息得到图像大小均值
    :param image_dict: roi兴趣图像
    :return: None
    """
    total_mean_list = []
    for key, values in image_dict.items():
        class_mean_list = []
        for i, image in enumerate(values):
            class_mean_list.append(image.shape[0:2])

        class_mean = tuple(map(lambda x: np.mean(x, dtype=np.int32), zip(*class_mean_list)))
        total_mean_list.append(class_mean)
        print(key, class_mean)

    print(f"总图像均值 {tuple(map(lambda x: np.mean(x, dtype=np.int32), zip(*total_mean_list)))}")


def image_random_resize(image, size_dict, min_size_threshold):

    assert sum(size_dict.values())-1 <= 9e-10, f"""
        size_dict的概率相加应当为1,但结果为 {sum(size_dict.values())}
    """

    p = np.array(list(size_dict.values()))

    resize = np.random.choice(list(size_dict.keys()), p=p.ravel())

    re_height, re_width, _ = image.shape

    if max(image.shape) > min_size_threshold:

        if re_height > re_width:
            re_width = re_width * (resize / re_height)
            re_height = resize
        else:
            re_height = re_height * (resize / re_width)
            re_width = resize

        # print(image.shape, (int(re_height), int(re_width)))

        return cv2.resize(image, (int(re_width), int(re_height)), interpolation=cv2.INTER_AREA)


def rotate(img, angle):

    height = img.shape[0]
    width = img.shape[1]

    if angle%180 == 0:
        scale = 1
    elif angle%90 == 0:
        scale = float(max(height, width))/min(height, width)
    else:
        scale = np.sqrt(pow(height, 2)+pow(width, 2))/min(height, width)

    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))

    return rotateImg


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))


class RandomImageGenerator:

    def __init__(self, image_dict: dict, background_list: list, label_dict: dict,
                 size_probability: dict, min_size_threshold: int=0):

        self.min_size_threshold = min_size_threshold
        self.image_dict = image_dict
        self.background_list = background_list
        self.label_dict = label_dict

        # # 如果没有，设置为概率平均 (废弃)
        # self.size_probability = size_probability if size_probability else\
        #     dict(
        #         zip(
        #             image_dict.keys(),
        #             [1 / image_dict.__len__()] * image_dict.__len__()
        #         )
        #     )
        self.size_probability = size_probability

        assert sum(self.size_probability.values()) - 1 <= 9e-10, f"""
                size_dict的概率相加应当为1,但结果为 {sum(self.size_probability.values())}
            """

        self.p = np.array(list(self.size_probability.values()))

        print(self.size_probability)

    def save(self, root_path, txt_path, total):
        f = open(txt_path, 'w')

        for label_name, image_list in self.image_dict.items():

            if total and total >= image_list.__len__():
                # print(label_name, image_list.__len__(), total // image_list.__len__(), total % image_list.__len__())
                image_list = image_list * (total // image_list.__len__())
                image_list.extend(image_list[:total % image_list.__len__()])

            label_path = os.path.join(root_path, label_name)
            if not os.path.exists(label_path):
                os.mkdir(label_path)

            # print(image_list.__len__())
            for i, image in enumerate(image_list):
                background = self.background_list[random.randint(0, self.background_list.__len__() - 1)].copy()

                resize = np.random.choice(list(self.size_probability.keys()), p=self.p.ravel())
                image = rotate_bound(image, random.randint(15, 135))

                re_height, re_width, _ = image.shape

                if max(image.shape) > self.min_size_threshold:

                    if re_height > re_width:
                        re_width = re_width * (resize / re_height)
                        re_height = resize
                    else:
                        re_height = re_height * (resize / re_width)
                        re_width = resize

                    # print(image.shape, (int(re_height), int(re_width)))

                    image = cv2.resize(image, (int(re_width), int(re_height)), interpolation=cv2.INTER_AREA)

                star_x = random.randint(0, background.shape[0]-image.shape[0])
                end_x = star_x + image.shape[0]

                star_y = random.randint(0, background.shape[1] - image.shape[1])
                end_y = star_y + image.shape[1]

                # 废弃
                # image = image.transpose((2, 0, 1))
                # image = scripted_transforms(torch.from_numpy(image))
                # image = image.numpy().transpose(1, 2, 0)

                # image[image == 0] = 255# re_background[star_x:end_x, star_y:end_y]
                # background[star_x:end_x, star_y:end_y] = image

                # image[image == 0] += background[star_x:end_x, star_y:end_y]
                roi = background[star_x:end_x, star_y:end_y]

                mask_inv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # mask = image.copy()
                mask_inv[mask_inv != 0] = 255

                # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # img_gray[img_gray != 255] = 0
                # ret, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # cv2.imshow('t', mask)

                mask = cv2.bitwise_not(mask_inv)
                roi_bg = cv2.bitwise_and(roi, roi, mask=mask)

                image_fg = cv2.bitwise_and(image, image, mask=mask_inv)
                dst = cv2.add(roi_bg, image_fg)

                background[star_x:end_x, star_y:end_y] = dst

                # cv2.imshow(f'{background.shape}', background)
                image_path = os.path.join(label_path, f"{i}.jpg")

                cv2.imencode('.jpg', background)[1].tofile(image_path)
                # cv2.imwrite(image_path, background)
                print(image_path)
                f.writelines(f"{image_path} {label_dict.get(label_name, None)}\n")
                # cv2.waitKey(100)
        f.close()


def label_txt(xml_info_list, txt_path):
    f = open(txt_path, 'w')
    temp_dict = dict()
    for i, info in enumerate(xml_info_list):
        if temp_dict.get(info['label'], None) is None:
            f.writelines(f"{info['label']} {temp_dict.__len__()}\n")
            temp_dict[info['label']] = temp_dict.__len__()

    f.close()
    return temp_dict


if __name__ == '__main__':
    xml_root_path = r"./markup_XML"
    image_root_path = r"./original_image"
    label_save_path = r"./label.txt"
    image_save_root = r"./train"
    every_class_image_num = 16
    txt_save_path = os.path.join(image_save_root, "train.txt")
    background_size = (512, 512)

    if not os.path.exists(image_save_root):
        os.mkdir(image_save_root)

    black_list = ['othercat']

    xml_info_list = read_xml_by_root(xml_root_path)

    print(xml_info_list)
    image_dict, background_list = split_roi_and_background(image_root_path, xml_info_list, black_list)
    label_dict = label_txt(xml_info_list, label_save_path)
    print(label_dict)
    # 可视化代码
    # for i, background in enumerate(background_list):
    #     cv2.imshow(f"bg{i}", background)
    #     cv2.waitKey(0)
    # for key, values in image_dict.items():
    #     for i, image in enumerate(values):
    #         cv2.imshow(f"{key},{i}", image)
    #         cv2.waitKey(0)
    #
    # cv2.waitKey(0)

    # 建议根据打印的情况选择resize roi区域的大小大小
    # 背景的话理论上也是干扰的一种，就不那么在意
    print_image_shape_mean(image_dict)

    background_list = list(map(lambda x: cv2.resize(x, background_size), background_list))

    # for i, background in enumerate(background_list):
    #     cv2.imshow(f"bg{i}", background)
    #     cv2.waitKey(0)

    '''
    我们数据集总均值为(847, 807)
    但其类别大小分布差距较大
    冒号 (973, 1154)
    屈屈 (1313, 1025)
    兔兔 (1216, 905)
    小美 (516, 693)
    小咪 (791, 697)
    丫头 (655, 718)
    雅雅 (374, 379)
    黛黛 (732, 519)
    警长 (758, 889)
    橘子皮 (1261, 1606)
    泡面 (443, 440)
    阿绿 (617, 361)
    骗子 (1362, 1115)
    
    我们的策略是选择最长的一边进行等比缩放，减少失真
    随后选择不同的比例缩放
    其依据是猫占图像比重 （0.6 -> 0.9）(1.0)
    此处选择了手动设置大小 背景512大小， 0.1占比大小为50
    我们更希望0.8为中心的数据集更多而小于310的图像原封不动
    占比   大小(max边) 概率(sum==1)
    0.6 -> 310          0.15
    0.7 -> 360          0.2
    0.8 -> 410          0.3
    0.9 -> 460          0.2
    1.0 -> 510          0.15
    '''

    # image_random_resize(
    #     image,
    #     size_dict={
    #         310: 0.15,
    #         360: 0.2,
    #         410: 0.3,
    #         460: 0.2,
    #         510: 0.15
    #     },
    #     min_size_threshold=310
    # )

    # for key, values in image_dict.items():
    #     for i, image in enumerate(values):
    #         print(image.shape)
    #         # cv2.imshow(f"{key},{i}", image)
    #         # cv2.waitKey(0)

    print(image_dict.__len__())

    size_dict = {
        310: 0.15,
        360: 0.2,
        410: 0.3,
        460: 0.2,
        510: 0.15
    }
    rig = RandomImageGenerator(image_dict, background_list, label_dict, size_dict)
    rig.save(image_save_root, txt_save_path, every_class_image_num)
