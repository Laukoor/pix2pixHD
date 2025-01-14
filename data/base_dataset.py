import torch.utils.data as data  # 导入 PyTorch 数据集模块，用于定义自定义数据集
from PIL import Image  # 导入 PIL 图像库，用于图像处理
import torchvision.transforms as transforms  # 导入 torchvision 的变换模块，用于图像预处理
import numpy as np  # 导入 NumPy，用于数值计算
import random  # 导入随机模块，用于随机化操作

# 定义 BaseDataset 类
class BaseDataset(data.Dataset):  # 继承 PyTorch 的 Dataset 基类
    def __init__(self):
        super(BaseDataset, self).__init__()  # 调用父类的初始化方法

    def name(self):  # 返回数据集名称
        return 'BaseDataset'

    def initialize(self, opt):  # 初始化方法，接收配置参数 opt（在子类中实现）
        pass

# 随机参数生成函数
# 用于生成随机的裁剪和翻转参数
# opt: 配置参数，size: 输入图像的尺寸 (宽, 高)
def get_params(opt, size):
    w, h = size  # 获取输入图像的宽度和高度
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':  # 如果启用了 resize_and_crop
        new_h = new_w = opt.loadSize  # 将图像调整为正方形
    elif opt.resize_or_crop == 'scale_width_and_crop':  # 如果启用了 scale_width_and_crop
        new_w = opt.loadSize  # 按目标宽度缩放
        new_h = opt.loadSize * h // w  # 按比例调整高度

    # 随机生成裁剪位置
    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    # 随机决定是否水平翻转
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}  # 返回裁剪位置和翻转标志

# 定义图像变换函数
# opt: 配置参数，params: 随机参数，method: 插值方式，normalize: 是否归一化
def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []  # 初始化变换列表
    if 'resize' in opt.resize_or_crop:  # 如果启用了 resize
        osize = [opt.loadSize, opt.loadSize]  # 目标尺寸
        transform_list.append(transforms.Resize(osize, method))  # 添加缩放操作
    elif 'scale_width' in opt.resize_or_crop:  # 如果启用了按宽度缩放
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:  # 如果启用了裁剪
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':  # 如果没有裁剪或缩放操作
        base = float(2 ** opt.n_downsample_global)  # 基础尺寸
        if opt.netG == 'local':  # 如果使用局部增强生成器
            base *= (2 ** opt.n_local_enhancers)  # 额外调整尺寸
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:  # 如果处于训练模式且允许翻转
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]  # 转换为 Tensor 格式

    if normalize:  # 如果启用归一化
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),  # 归一化均值
                                                (0.5, 0.5, 0.5))]  # 归一化标准差
    return transforms.Compose(transform_list)  # 返回组合的变换

# 返回固定的归一化操作（均值为 0.5，标准差为 0.5）
def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# 调整尺寸为 2 的幂次方
# img: 输入图像，base: 基础尺寸，method: 插值方式
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size  # 获取原始宽高
    h = int(round(oh / base) * base)  # 调整高度为基数的整数倍
    w = int(round(ow / base) * base)  # 调整宽度为基数的整数倍
    if (h == oh) and (w == ow):  # 如果宽高已满足条件，则直接返回
        return img
    return img.resize((w, h), method)  # 否则调整图像尺寸

# 按宽度缩放
# img: 输入图像，target_width: 目标宽度，method: 插值方式
def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size  # 获取原始宽高
    if (ow == target_width):  # 如果宽度已经是目标宽度，直接返回
        return img
    w = target_width  # 目标宽度
    h = int(target_width * oh / ow)  # 按比例调整高度
    return img.resize((w, h), method)  # 调整图像尺寸

# 裁剪图像到指定区域
# img: 输入图像，pos: 裁剪位置，size: 裁剪尺寸
def __crop(img, pos, size):
    ow, oh = img.size  # 获取原始宽高
    x1, y1 = pos  # 裁剪位置
    tw = th = size  # 裁剪区域大小
    if (ow > tw or oh > th):  # 如果图像尺寸大于裁剪区域
        return img.crop((x1, y1, x1 + tw, y1 + th))  # 按位置裁剪
    return img  # 否则返回原图

# 随机水平翻转
# img: 输入图像，flip: 翻转标志
def __flip(img, flip):
    if flip:  # 如果 flip 为 True
        return img.transpose(Image.FLIP_LEFT_RIGHT)  # 进行水平翻转
    return img  # 否则返回原图
