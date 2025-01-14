from .base_options import BaseOptions  # 导入 BaseOptions 类

# 定义训练选项类，继承自 BaseOptions
class TrainOptions(BaseOptions):
    def initialize(self):  
        BaseOptions.initialize(self)  # 调用父类的 initialize 方法，加载基础选项

        # 显示选项（控制训练过程中输出显示的频率等）
        self.parser.add_argument('--display_freq', type=int, default=100, 
                                 help='frequency of showing training results on screen')  
                                 # 控制显示训练结果到屏幕的频率
        self.parser.add_argument('--print_freq', type=int, default=100, 
                                 help='frequency of showing training results on console')  
                                 # 控制输出训练结果到控制台的频率
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, 
                                 help='frequency of saving the latest results')  
                                 # 保存最新模型的频率
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, 
                                 help='frequency of saving checkpoints at the end of epochs')  
                                 # 每隔多少个 epoch 保存一次检查点
        self.parser.add_argument('--no_html', action='store_true', 
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')  
                                 # 如果指定此选项，不保存训练中间结果到 HTML 文件
        self.parser.add_argument('--debug', action='store_true', 
                                 help='only do one epoch and displays at each iteration')  
                                 # 如果启用调试模式，只运行一个 epoch，并在每次迭代后显示结果

        # 训练相关选项
        self.parser.add_argument('--continue_train', action='store_true', 
                                 help='continue training: load the latest model')  
                                 # 是否继续训练，加载最新保存的模型
        self.parser.add_argument('--load_pretrain', type=str, default='', 
                                 help='load the pretrained model from the specified location')  
                                 # 从指定路径加载预训练模型
        self.parser.add_argument('--which_epoch', type=str, default='latest', 
                                 help='which epoch to load? set to latest to use latest cached model')  
                                 # 指定要加载的模型的 epoch
        self.parser.add_argument('--phase', type=str, default='train', 
                                 help='train, val, test, etc')  
                                 # 指定训练阶段的名称，默认为 "train"
        self.parser.add_argument('--niter', type=int, default=100, 
                                 help='# of iter at starting learning rate')  
                                 # 在初始学习率下训练的迭代次数
        self.parser.add_argument('--niter_decay', type=int, default=100, 
                                 help='# of iter to linearly decay learning rate to zero')  
                                 # 学习率线性衰减到 0 的迭代次数
        self.parser.add_argument('--beta1', type=float, default=0.5, 
                                 help='momentum term of adam')  
                                 # Adam 优化器的动量参数 β1
        self.parser.add_argument('--lr', type=float, default=0.0002, 
                                 help='initial learning rate for adam')  
                                 # Adam 优化器的初始学习率

        # 判别器相关选项
        self.parser.add_argument('--num_D', type=int, default=2, 
                                 help='number of discriminators to use')  
                                 # 使用的判别器数量
        self.parser.add_argument('--n_layers_D', type=int, default=3, 
                                 help='only used if which_model_netD==n_layers')  
                                 # 判别器网络中卷积层的数量（仅当判别器模型是多层时使用）
        self.parser.add_argument('--ndf', type=int, default=64, 
                                 help='# of discrim filters in first conv layer')  
                                 # 判别器第一卷积层的过滤器数量
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, 
                                 help='weight for feature matching loss')  
                                 # 特征匹配损失的权重
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', 
                                 help='if specified, do *not* use discriminator feature matching loss')  
                                 # 如果指定此选项，不使用判别器的特征匹配损失
        self.parser.add_argument('--no_vgg_loss', action='store_true', 
                                 help='if specified, do *not* use VGG feature matching loss')  
                                 # 如果指定此选项，不使用 VGG 特征匹配损失
        self.parser.add_argument('--no_lsgan', action='store_true', 
                                 help='do *not* use least square GAN, if false, use vanilla GAN')  
                                 # 如果指定此选项，不使用最小平方 GAN 损失，改用标准 GAN 损失
        self.parser.add_argument('--pool_size', type=int, default=0, 
                                 help='the size of image buffer that stores previously generated images')  
                                 # 图像缓冲区的大小，用于存储之前生成的图像

        self.isTrain = True  # 设置 isTrain 为 True，表示这是训练模式
