import sys

args_list = sys.argv[1:]

"""
获取参数中的赋值操作
并以字节码运行

python args.py test=1 test2=2
意味着变量test的值为1 变量test2的值为2
如果程序没有定义此变量，则创建并赋值
如果有此变量则是正常的赋值语句

任意文件可以引用此文件
但需要注意的是！：此操作有安全风险，请自行判断是否为安全的环境
"""


def get_args_compile():
    args_str = ""
    for args in args_list:
        assert args.find("=") != -1, f"""
            格式应当为： var1=1 var2=2
            意为： 变量var1的值为1 变量var2的值为2
            报错内容：[{args}] 并没有'='
            单个值间用等号相连，各个值间用空格相隔
        """
        args_str += f"{args}\n"

    return compile(args_str, '', 'exec')


# exec(get_args_compile())
