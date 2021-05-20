from .models import *
# from .deform_conv import DeformConv2D  # 已经废弃，其使用方法如同torchvision示例：被我迁移至新库
# from .vision_deform_conv import DeformConv2d_v1, DeformConv2d_v2  # 被嵌入模型中
# DeformConv2D未来会在Torchvision.ops.DeformConv2D中选择

"""
torchvision.ops.DeformConv2D(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True)
 
 offset_net = nn.Conv2d(512, 2*3*3， 3）
 dcnv2 = t.ops.dcm(512, 512, 3) - >k*k*3
"""

"""
deform_conv.DeformConv2D (inc, outc, kernel_size=3, padding=1, bias=None)

"""

"""
vision_deform_conv -> v1、2并非dcn与dcnv2的区别，而是dcnv2的实现不同！
v1 使用torchvision 建议使用！在其中提供offset，无需额外提供

v2 使用 deform_conv中，修改无需在前向中额外提供offset，offset提供在模型中，是由官方库的第三方复现后修改 不推荐！
modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).

"""