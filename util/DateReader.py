from functools import wraps
from queue import Queue
import threading
import random
import sys

import cv2
import numpy as np

DEBUG = False


def log_print(*args, **kwargs):
    """
    注: 在cmd等不支持彩色的命令框 log_print函数无法发挥他颜色的效果
    :param args: 一帆风顺
    :param kwargs:  一帆风顺
    :return:
    """
    print("\033[0;31m[log:\033[0m", *args, "\033[0;31m]\033[0m", **kwargs)


def data_read_function(debug: bool=False):
    def decorator(func):
        def _check_func(*args, **kwargs):
            assert func.__code__.co_argcount - 1 >= 0, """
                被data_read_function修饰的函数应当有1个以上参数
            """

            assert len(args)+len(kwargs)+1 == func.__code__.co_argcount, f"""
                输入应当是{func.__code__.co_argcount - 1}个参数，但是输入了{len(args)+len(kwargs)}个
                args:{str(args)}, kwargs:{str(kwargs)}
                可能是: 
                    被data_read_function修饰且debug为False的函数在第一次传参时仅需要传
                除去第一个位置参数外的其他参数，并且形式不限,请注意是否额外传入了第一个参数
                (因为按照模板，第一个参数应当是image(变量名可改),此参数将在参与运算时自动call)
                e.g: def test(image, size) using: test(1)(2)  则size = 1 , image = 2
                请不要如此操作test(1,3)(2),这将引发此类错误
            """

            assert not kwargs.get(func.__code__.co_varnames[0]), f"""
                    第一个参数 {func.__code__.co_varnames[0]} 将会在参与运算时自动call
                所以你不应当显式的指定他的值
                e.g: def test(image, size) using: test(1)(2)  则size = 1 , image = 2
                请不要如此操作test(image=1)(2),这将引发此类错误
            """

            def _temp_func(*args, **kwargs):
                def _call_func(img):
                    return func(img, *args, **kwargs)

                return wraps(func)(_call_func)
            return _temp_func(*args, **kwargs)

        if debug is True:
            # 此处is True 并不是多余的
            # 是debug模式，直接返回原函数，不做修改
            return func

        # 非debug时,修饰检查并构造我们想要的函数
        return _check_func

    # if isinstance(debug, types.FunctionType):
    if callable(debug):
        # 此时@data_read_function没有( )被调用
        # debug 是 被修饰函数
        return decorator(debug)

    # 此时@data_read_function有( )被调用
    # debug 按照提示应当是bool
    return decorator


"""
data_read_function修饰器介绍:
被data_read_function修饰器修饰会有如下效果
    1.允许在传入函数名时进行函数调用
    2.非第一个参数外的所有参数并不会多次传入
被data_read_function修饰器修饰会有如下要求
    1. 被data_read_function修饰的函数应当有1个以上参数
    2. 第一个参数预设是用来接受图像或者其他预定义作用的
    3. 在使用被修饰函数时，不需要提供第一个参数，第一个
    参数将会在使用此方法的时候自动传入

data_read_function修饰时
可以使用@data_read_function或者@data_read_function(debug)

使用后者可以传入一个参数:是否是debug模式,默认是False
@data_read_function(debug=True)
@data_read_function(True)

使用debug模式data_read_function修饰器将失效
函数的使用方法回归平常

e.g:
    @data_read_function
    def test(img, r):
        print(f"img{img}, r{r}")
        return 1
    
print(test(2)(1))
或者print(test(r=2)(1))
输出：
    >>> img1, r2
    >>> 1
    
被修饰的函数test使用方法从
test(1,2)
改编为
test(2)(1)
他们是等效的
其中唯一不同是
2这个参数在test(2)(1)每次被调用时不会多次传入

如果你不想用修饰器，那么可以使用闭包
但请同样遵守第一个参数为图像接收参数
e.g:
    def img_resize(size):
        def resize(img):
            print(img, size)
            return 1
        return wraps(img_resize)(resize)
    
print(img_resize(2)(1))
效果和上面相似

"""


class BasicDataReader(object):

    def __init__(self):
        pass

    def read_data(self, img_path):

        raise RuntimeError("read_data not define")

    def read_txt(self, txt_path):

        raise RuntimeError("read_txt not define")

    @staticmethod
    def show_progress(total: int) -> str:
        """
        进度条，迭代器每一次迭代前进 1/total
        e.g:它来自以前的代码
        :param total: 总数
        :return: 进度条字符串生成器
        """
        while True:
            for num in range(1, total + 1):
                print_num = num * (50 / total)
                print_str = '[%-50s] %s/%s ' % ('=' * int(print_num), str(num), str(total))
                yield print_str


class CacheList(list):

    def __init__(self, data_cache: int=0):
        super(CacheList, self).__init__()
        self.data_cache = data_cache if data_cache > 0 else 0

    @property
    def is_full(self):
        return self.__len__() >= self.data_cache and self.data_cache

    def append(self, data: object):
        if self.__len__() >= self.data_cache and self.data_cache:
            self.pop(0)

        super(CacheList, self).append(data)


class DataReader(BasicDataReader):

    def __init__(self, image_txt_path: str, function_list: list,
                 batch_size: int, *, read_data_func=None,
                 read_txt_func=None, is_show_progress: bool=True,
                 read_data_cache: int=0, is_completion: bool=True,
                 is_shuffle: bool=True, using_thread_num: int=1):
        """

        :param image_txt_path: 存放图像地址的txt文件路径

        :param function_list:  需要预处理的函数列表，他在数据读取时起作用
        注意： 存在此列表中的函数应当是data_read_function修饰或者闭包的函数，请看上方注释

        :param batch_size: 一批数据的大小

        :param read_data_func: 读取图像数据的函数，定义形式应当包含一个参数用于接收传入字符串
        为None时默认使用DataReader自带的读取方式，即opencv imread,且不做其他处理，这通常是用来读取多个图片
        如果你需要读取图片+标签的形式， 那么使用自定义的函数或是预设定的one_hot_label_read_func和label_read_func

        :param read_txt_func:  读取txt文件的函数，他是读取图像地址的函数，定义形式应当包含一个参数用于接收传入字符串
        为None是默认使用DataReader自带的读取方式，通过行读取，空格等分割得到结果，默认读取时单行的数目可以是不固定的

        :param is_show_progress: 是否展示数据读取的进度条

        :param read_data_cache: 数据缓冲区大小，为零意味着不设定缓冲，将全部数据一次读取

        注： 建议为批大小的倍数, 当is_completion为True时，缓冲区大小并非由输入决定
        而由缓冲同批大小补全决定

        e.g: read_data_cache:20, is_completion:True,  batch_size:16 那么read_data_cache被补全:32

        :param is_completion: 是否补全文件数为批大小的倍数

        注：此选项为True会改变文件数目,并且多余补偿部分是来自文件本身
        同样的，缓冲区也将被补全
        如果不补全，请注意缓冲区大于批且不能整除的情况，这会导致丢失掉部分缓冲区内的数据（未来版本可能改进）
        e.g: read_data_cache:19, batch_size:16 那么一批数据19中仅仅有16个数据被使用，后面的3个被丢弃
        """

        super(DataReader, self).__init__()
        self.read_data = self.read_data if not read_data_func else read_data_func
        self.read_txt = self.read_txt if not read_txt_func else read_txt_func

        self.is_show_progress: bool = is_show_progress
        self.is_completion: bool = is_completion

        self.batch_size: int = int(batch_size) if batch_size >= 0 else 1
        self.is_shuffle: bool = is_shuffle
        self.ont_epoch_step: int = 1
        self.step: int = 0
        self.epoch: int = 0

        self._txt_content = self.read_txt(image_txt_path)

        self.batch_size = self.batch_size if self.batch_size else self._txt_content.__len__()

        assert self._txt_content.__len__() >= self.batch_size,\
            f"""
                    批大小应当小于等于数据总量
                    请检查TXT中是否总量达不到批大小
                    TXT中读取到的总量是：{self._txt_content.__len__()}
                    批大小为：{self.batch_size}
             """

        # 临时变量 ： 计算读取到的总文件数是否能被批大小整除
        _temp = self._txt_content.__len__() % self.batch_size
        # 如果is_completion（是否补全）为true且不能整除
        if self.is_completion and _temp != 0:
            # 取开头 批大小到余数的差值 进行补全  可进一步修改随机取
            self._txt_content.extend(self._txt_content[: self.batch_size - _temp])
            log_print(f"is_completion为{self.is_completion},"
                      f"原数量{self._txt_content.__len__() - (self.batch_size - _temp)},"
                      f"文件数量补全为{self._txt_content.__len__()}")

        # total总数等于补全后或者原文件的数量
        self.total = self._txt_content.__len__()
        if self.is_shuffle:
            random.shuffle(self._txt_content)

        # 读取的缓冲区大小如果为空或者0，就选择全部读取，即大小为总数
        self.read_data_cache = read_data_cache if read_data_cache else self.total

        # 断言 缓冲区的大小应当大于等于批大小，否则效率太低，如果需要去掉此限制
        # 需要修改__getitem__函数，实现多次获取缓冲区内容一次输出的方法
        assert self.read_data_cache >= self.batch_size, f"""
            缓冲区大小：{self.read_data_cache} 应当大于等于批大小：{self.batch_size}
        """

        # 临时变量 ： 计算缓冲区大小是否能被批大小整除
        _temp = self.read_data_cache % self.batch_size

        # 如果缓冲区大小不能被整除且is_completion为ture,进行补全
        if self.is_completion and _temp != 0:
            self.read_data_cache += self.batch_size - _temp
            log_print(f"is_completion为{self.is_completion},缓冲区补全为{self.read_data_cache}")

        # 断言文件数是否大于等于读取缓冲区数
        # 如果此断言不存在，那么可能出现缓冲区永远等不到数据读取完成
        assert self._txt_content.__len__() >= self.read_data_cache, \
            f"""
                    缓冲区大小同样应当小于等于总量
                    请检查TXT中是否总量达不到批大小
                    TXT中读取到的总量是：{self._txt_content.__len__()}
                    缓冲区大小：{self.read_data_cache}
                    注意：当较低数量的文件总数，批大小不能整除
                    触发补全机制时，缓冲区是暂时允许超过 原 文件总数的
             """

        # bool类型 缓冲区大小是否是总数
        self._cache_is_total: bool = self.read_data_cache == self.total
        # total_step 单批总迭代步数 他被__getitem__中用来判断是否完成一次总数的迭代
        self.total_step = self.total // self.batch_size if self._cache_is_total \
            else self.read_data_cache // self.batch_size

        using_thread_num = int(using_thread_num) if using_thread_num else 1
        # 数据获取迭代器，每次调用迭代器，就会获得一定的数据，这取决于你缓冲区大小
        self.data_iter = self.get_data_iter(function_list=function_list, using_thread_num=using_thread_num)

    def read_txt(self, txt_path):
        """
        读取txt方式函数
        :param txt_path: TXT目录地址
        :return: list [ [str, ...], [str, ...], ...]
        """
        with open(txt_path) as f_read:
            _txt_content = f_read.readlines()

        return [path.split() for path in _txt_content]

    def read_data(self, img_path):
        """
        读取数据方式函数
        :param img_path:  图像地址
        :return: ndarray
        """
        return cv2.imread(img_path)

    def thread_worker(self, product_queue, consume_queue, function_list, progress):
        """
        线程工作者
        :param product_queue:
        :param consume_queue:
        :param function_list:
        :param progress:
        :return:
        """
        while True:
            _txt_content = product_queue.get()
            read_data_tuple = tuple(map(self.read_data, _txt_content))

            for func in function_list:
                read_data_tuple = tuple(map(func, read_data_tuple))

            consume_queue.put(read_data_tuple)

            if self.is_show_progress:
                sys.stdout.flush()
                sys.stdout.write('\r' + next(progress))

    def product_put_worker(self, product_queue):
        """
        数据提交者
        :param product_queue:
        :return:
        """
        while True:
            if self.is_shuffle:
                random.shuffle(self._txt_content)
            for i in range(self._txt_content.__len__()):
                product_queue.put(self._txt_content[i])
            if self._cache_is_total:
                break

    def get_data_iter(self, function_list, using_thread_num):
        """
        :param function_list: 需要执行的函数列表
        其应当遵守:
            使用data_read_function修饰器或者闭包或函数名
            具体请参加data_read_function详解
        :param using_thread_num: 生产者线程数量
        :return:
        """

        product_queue = Queue(maxsize=self.read_data_cache)
        consume_queue = Queue(maxsize=self.read_data_cache)

        cache_data = CacheList(self.read_data_cache)
        progress = self.show_progress(self._txt_content.__len__())

        put_worker = threading.Thread(target=self.product_put_worker, args=(product_queue,))
        put_worker.setDaemon(True)

        for i in range(using_thread_num):
            product_worker = threading.Thread(target=self.thread_worker, args=(product_queue, consume_queue,
                                                                               function_list, progress))
            product_worker.setDaemon(True)
            product_worker.start()

        put_worker.start()

        while True:
            if consume_queue.full():
                for i in range(self.read_data_cache):
                    cache_data.append(consume_queue.get())

                yield tuple(zip(*cache_data))

    def __getitem__(self, item):
        _temp_step = self.step
        # 第一次迭代统一获取数据
        if _temp_step == 0:
            self.item_data = self.data_iter.__next__()

        if _temp_step > self.total_step - 1:
            _temp_step = self.step = 0

            # 如果是总数，那么没必要再次获取
            if not self._cache_is_total:
                self.item_data = self.data_iter.__next__()

        out_data = [data[self.batch_size * _temp_step: self.batch_size * (_temp_step + 1)] for data in self.item_data]

        # 此处判断是否完成一批所需要的step，请注意这里不能替换成self.total_step
        # 因为前者会因为缓冲区大小而改变，这里只能是总数
        if self.ont_epoch_step - 1 >= self.total // self.batch_size:
            self.ont_epoch_step = 1
            self.epoch += 1

        self.step += 1
        self.ont_epoch_step += 1

        return (self.epoch,  *out_data)


# @data_read_function(DEBUG)  # 是否使用修饰都可以
def img_padding(image):
    if not is_image(image):
        return image

    width, high, channel = image.shape if len(image.shape) == 3 else (*image.shape, 1)

    dim_diff = np.abs(high - width)

    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if high <= width else (pad1, pad2, 0, 0)
    top, bottom, left, right = pad

    img_pad = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

    return img_pad


def is_image(image):
    return image.shape.__len__() in (2, 3)


@data_read_function(DEBUG)
def img_resize(image, resize):
    if not is_image(image):
        return image

    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

    return image


@data_read_function(DEBUG)
def gray_img_reshape(image):
    if not is_image(image):
        return image

    if len(image.shape) == 2:
        image = image.reshape((image.shape[0], image.shape[1], 1))

    return image


@data_read_function(DEBUG)
def img_BGR2YUV(image):
    if not is_image(image):
        return image

    if not img_check_gray(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    return image


@data_read_function(DEBUG)
def img_random_resize_from_dict(image, size_dict):
    if not is_image(image):
        return image

    assert sum(size_dict.values()) == 1.0, "using_mode_dict的概率相加应当为1"

    p = np.array(list(size_dict.values()))

    resize = np.random.choice(list(size_dict.keys()), p=p.ravel())
    re_img = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_AREA)

    return re_img


@data_read_function(DEBUG)
def img_float_normalization(image, max_float=255.0):
    if not is_image(image):
        return image

    return image.astype('float32') / max_float


def img_check_gray(image):
    if not is_image(image):
        return image

    total = image.shape[0] * image.shape[1]

    image = image.reshape(-1, 3)
    image = image[image[:, 0] == image[:, 1]]
    image = image[image[:, 1] == image[:, 2]]

    return len(image) == total


def one_hot_by_container(labels_list, label_class):
    """
    将列表、元祖等容器转换成一热编码
    :param labels_list:  list or tuple
    :param label_class: 种类数量
    :return:np.ndarray
    """
    return np.array([[int(i == int(labels_list[j])) for i in range(label_class)] for j in range(len(labels_list))])


def one_hot(labels, label_class):
    """
    将单个标签转换成一热编码
    :param labels: int or str
    :param label_class: 种类数量
    :return: np.ndarray
    """

    return np.array([int(i == int(labels)) for i in range(label_class)])


@data_read_function(DEBUG)
def one_hot_label_read_func(img_path, label_class):
    if '.' in img_path:
        return cv2.imread(img_path)

    return one_hot(img_path, label_class)


def label_read_func(img_path):
    if '.' in img_path:
        return cv2.imread(img_path)

    return int(img_path)


if __name__ == '__main__':

    dr = DataReader(r"test.txt", [
            img_padding,  # img_padding没使用修饰，所以传入函数名，如果使用修饰此处使用img_padding()
            img_resize((512, 512)),
            img_BGR2YUV(),
        ],
        read_data_func=one_hot_label_read_func(3),
        read_data_cache=20,  # 读取数据的缓冲大小
        batch_size=10,  # 读取数据得到的批大小 必填
        is_completion=True,  # 是否填充
        using_thread_num=5,  # 使用线程数
        is_show_progress=True  # 是否可视化读取进程
        # read_txt_func  # 读取文本的函数 使用默认
        # is_shuffle  # 是否乱序 使用默认
    )

    for epoch, image, label in dr:
        image = np.array(image).astype("float32") / 255.0
        label = np.array(label)

        print(image.shape, label.shape, epoch)

