###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os

# 定义支持的图像文件扩展名列表
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

# 检查文件是否是支持的图像格式
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 遍历目录及其子目录，生成所有图像文件的路径列表
def make_dataset(dir):
    images = []  # 初始化一个空列表用于存储图像路径
    assert os.path.isdir(dir), '%s is not a valid directory' % dir  # 检查路径是否合法

    # 使用 os.walk 遍历目录
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:  # 遍历文件
            if is_image_file(fname):  # 检查文件是否是图像
                path = os.path.join(root, fname)  # 获取文件完整路径
                images.append(path)  # 添加到列表

    return images  # 返回图像路径列表

# 加载图像并将其转换为 RGB 格式
def default_loader(path):
    return Image.open(path).convert('RGB')

# 定义自定义数据集类 ImageFolder
class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        """
        初始化 ImageFolder 数据集。

        参数：
        - root: 图像数据所在的根目录。
        - transform: 可选的图像变换操作。
        - return_paths: 是否返回图像路径。
        - loader: 用于加载图像的函数，默认为 default_loader。
        """
        imgs = make_dataset(root)  # 获取图像路径列表
        if len(imgs) == 0:  # 如果没有找到图像，抛出异常
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root  # 保存根目录路径
        self.imgs = imgs  # 保存图像路径列表
        self.transform = transform  # 保存图像变换操作
        self.return_paths = return_paths  # 保存是否返回路径的选项
        self.loader = loader  # 保存加载图像的函数

    def __getitem__(self, index):
        """
        获取数据集中的第 index 个样本。

        参数：
        - index: 数据索引。

        返回：
        - 图像张量，如果 return_paths=True，则还会返回图像路径。
        """
        path = self.imgs[index]  # 根据索引获取图像路径
        img = self.loader(path)  # 使用 loader 加载图像
        if self.transform is not None:  # 如果定义了变换操作
            img = self.transform(img)  # 对图像进行变换
        if self.return_paths:  # 如果需要返回路径
            return img, path  # 返回图像和路径
        else:
            return img  # 仅返回图像

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.imgs)  # 返回图像路径列表的长度
