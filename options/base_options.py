import argparse
import os
from util import util  # 引入工具类，用于创建目录等操作
import torch

class BaseOptions():
    def __init__(self):
        # 初始化 argparse 解析器
        self.parser = argparse.ArgumentParser()
        self.initialized = False  # 表示选项尚未初始化

    def initialize(self):    
        # 实验相关参数
        self.parser.add_argument('--name', type=str, default='label2city', 
                                 help='实验名称，用于决定保存样本和模型的位置')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', 
                                 help='GPU IDs，例如 0 或 0,1,2，使用 -1 表示 CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                                 help='模型保存的路径')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', 
                                 help='使用的模型类型')
        self.parser.add_argument('--norm', type=str, default='instance', 
                                 help='选择归一化类型：instance normalization 或 batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', 
                                 help='是否在生成器中使用 Dropout')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], 
                                 help="支持的数据类型，例如 8、16 或 32 位")
        self.parser.add_argument('--verbose', action='store_true', default=False, 
                                 help='是否启用详细模式')
        self.parser.add_argument('--fp16', action='store_true', default=False, 
                                 help='是否启用混合精度训练 (AMP)')
        self.parser.add_argument('--local_rank', type=int, default=0, 
                                 help='分布式训练时的本地设备序号')

        # 输入/输出尺寸相关参数
        self.parser.add_argument('--batchSize', type=int, default=1, 
                                 help='输入的批量大小')
        self.parser.add_argument('--loadSize', type=int, default=1024, 
                                 help='将图像缩放到此大小')
        self.parser.add_argument('--fineSize', type=int, default=512, 
                                 help='然后裁剪到此大小')
        self.parser.add_argument('--label_nc', type=int, default=35, 
                                 help='输入标签的通道数')
        self.parser.add_argument('--input_nc', type=int, default=3, 
                                 help='输入图像的通道数')
        self.parser.add_argument('--output_nc', type=int, default=3, 
                                 help='输出图像的通道数')

        # 输入相关设置
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/', 
                                 help='数据路径')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', 
                                 help='图像加载时的缩放和裁剪方式 [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', 
                                 help='是否按顺序生成批次数据，否则随机采样')
        self.parser.add_argument('--no_flip', action='store_true', 
                                 help='如果指定，则不进行水平翻转的数据增强') 
        self.parser.add_argument('--nThreads', default=2, type=int, 
                                 help='加载数据的线程数量')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), 
                                 help='每个数据集允许的最大样本数，超出则截取子集')

        # 显示相关参数
        self.parser.add_argument('--display_winsize', type=int, default=512,  
                                 help='显示窗口的大小')
        self.parser.add_argument('--tf_log', action='store_true', 
                                 help='是否使用 TensorBoard 进行日志记录（需要安装 TensorFlow）')

        # 生成器相关参数
        self.parser.add_argument('--netG', type=str, default='global', 
                                 help='选择生成器模型')
        self.parser.add_argument('--ngf', type=int, default=64, 
                                 help='生成器第一层卷积滤波器数量')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, 
                                 help='生成器中下采样层的数量') 
        self.parser.add_argument('--n_blocks_global', type=int, default=9, 
                                 help='全局生成器网络中的残差块数量')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, 
                                 help='局部增强生成器网络中的残差块数量')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, 
                                 help='使用的局部增强器数量')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, 
                                 help='仅训练局部增强器的 epoch 数量')        

        # 实例特征相关参数
        self.parser.add_argument('--no_instance', action='store_true', 
                                 help='如果指定，则不添加实例分割图作为输入')        
        self.parser.add_argument('--instance_feat', action='store_true', 
                                 help='如果指定，添加编码的实例特征作为输入')
        self.parser.add_argument('--label_feat', action='store_true', 
                                 help='如果指定，添加编码的标签特征作为输入')        
        self.parser.add_argument('--feat_num', type=int, default=3, 
                                 help='特征向量的长度')        
        self.parser.add_argument('--load_features', action='store_true', 
                                 help='如果指定，加载预计算的特征图')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, 
                                 help='编码器中的下采样层数量') 
        self.parser.add_argument('--nef', type=int, default=16, 
                                 help='编码器第一层卷积的滤波器数量')        
        self.parser.add_argument('--n_clusters', type=int, default=10, 
                                 help='特征的聚类数量')        

        self.initialized = True  # 表示选项已经初始化

    def parse(self, save=True):
        # 解析选项
        if not self.initialized:
            self.initialize()  # 初始化选项
        self.opt = self.parser.parse_args()  # 解析命令行参数
        self.opt.isTrain = self.isTrain  # 设置是训练还是测试模式

        # 解析 GPU ID
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # 设置 GPU
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # 打印选项
        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # 保存选项到磁盘
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)  # 创建实验目录
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt  # 返回解析结果
