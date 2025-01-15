from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import os
import util.util as util
from torch.autograd import Variable
import torch.nn as nn

opt = TrainOptions().parse()
opt.nThreads = 1              # 数据加载线程数设置为 1
opt.batchSize = 1             # 每次处理一张图像
opt.serial_batches = True     # 按顺序加载数据
opt.no_flip = True            # 禁用随机水平翻转
opt.instance_feat = True      # 启用实例级特征

name = 'features'
save_path = os.path.join(opt.checkpoints_dir, opt.name)

############ Initialize #########
data_loader = CreateDataLoader(opt)  # 创建数据加载器
dataset = data_loader.load_data()    # 加载数据集
dataset_size = len(data_loader)      # 数据集大小
model = create_model(opt)            # 创建模型
util.mkdirs(os.path.join(opt.dataroot, opt.phase + '_feat'))  # 创建特征图保存目录

######## Save precomputed feature maps for 1024p training #######
for i, data in enumerate(dataset):  # 遍历数据集中的每张图片
    print('%d / %d images' % (i+1, dataset_size))  # 打印进度

    # 使用模型的编码器（netE）提取特征图
    feat_map = model.module.netE.forward(
        Variable(data['image'].cuda(), volatile=True),  # 输入真实图像
        data['inst'].cuda()                            # 输入实例图
    )

    # 将特征图上采样（倍增分辨率）
    feat_map = nn.Upsample(scale_factor=2, mode='nearest')(feat_map)

    # 将特征图转换为 NumPy 格式
    image_numpy = util.tensor2im(feat_map.data[0])

    # 修改路径，将特征图保存到指定位置
    save_path = data['path'][0].replace('/train_label/', '/train_feat/')
    util.save_image(image_numpy, save_path)
