import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

# 定义 Pix2PixHD 模型类
class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'  # 返回模型名称

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        # 初始化损失过滤器，动态控制哪些损失函数被使用
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)  # 各种损失的标志位
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            # 返回仅启用的损失
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)  # 调用父类的初始化方法
        if opt.resize_or_crop != 'none' or not opt.isTrain:
            # 若启用全分辨率训练，优化 cudnn 性能
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain  # 是否处于训练模式
        self.use_features = opt.instance_feat or opt.label_feat  # 是否使用特征
        self.gen_features = self.use_features and not self.opt.load_features  # 是否生成特征
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc  # 输入通道数

        ##### 定义网络 #####
        # 定义生成器网络
        netG_input_nc = input_nc  # 生成器输入通道数
        if not opt.no_instance:  # 如果需要实例信息
            netG_input_nc += 1
        if self.use_features:  # 如果使用特征
            netG_input_nc += opt.feat_num
        # 初始化生成器
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # 定义判别器网络
        if self.isTrain:
            use_sigmoid = opt.no_lsgan  # 是否使用 sigmoid 激活
            netD_input_nc = input_nc + opt.output_nc  # 判别器输入通道为输入与输出的拼接
            if not opt.no_instance:  # 包含实例信息
                netD_input_nc += 1
            # 初始化判别器
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### 定义编码器网络
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # 加载预训练模型
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)  # 加载生成器
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  # 加载判别器
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)  # 加载编码器

        # 设置损失函数和优化器
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                # 如果使用多 GPU，不支持 Fake Pool
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)  # 初始化假样本池
            self.old_lr = opt.lr  # 保存学习率

            # 定义损失函数
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)  # 初始化损失过滤器
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)  # 定义 GAN 损失
            self.criterionFeat = torch.nn.L1Loss()  # 定义 L1 损失
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)  # 定义 VGG 损失

            # 定义损失名称
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')

            # 初始化优化器
            # 优化器 G
            if opt.niter_fix_global > 0:  # 固定全局生成器部分
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():  # 遍历生成器参数
                    if key.startswith('model' + str(opt.n_local_enhancers)):  # 只训练局部增强生成器
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())  # 获取所有生成器参数
            if self.gen_features:  # 如果需要生成特征
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))  # Adam 优化器

            # 优化器 D
            params = list(self.netD.parameters())  # 获取判别器参数
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))  # Adam 优化器

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        # 编码输入数据
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()  # 如果没有标签数，直接返回输入
        else:
            # 创建标签的 one-hot 表示
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)  # 转换为 one-hot
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # 从实例图提取边缘信息
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)  # 拼接边缘图
        input_label = Variable(input_label, volatile=infer)  # 转换为变量

        # 训练用的真实图片
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # 编码特征图
        if self.use_features:
            if self.opt.load_features:  # 如果加载预训练特征
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        # 判别输入图像
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)  # 拼接输入
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)  # 查询假样本池
            return self.netD.forward(fake_query)  # 判别
        else:
            return self.netD.forward(input_concat)  # 判别

    def forward(self, label, inst, image, feat, infer=False):
        # 编码输入
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)

        # 生成伪造图片
        if self.use_features:  # 如果使用特征
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)  # 使用编码器提取特征
            input_concat = torch.cat((input_label, feat_map), dim=1)  # 将标签和特征拼接
        else:
            input_concat = input_label
        fake_image = self.netG.forward(input_concat)  # 生成伪造图片

        # 判别伪造图片并计算损失
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)  # 使用假样本池
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)  # 判别伪造样本的损失

        # 判别真实图片并计算损失
        pred_real = self.discriminate(input_label, real_image)  # 判别真实样本
        loss_D_real = self.criterionGAN(pred_real, True)  # 判别真实样本的损失

        # 计算生成器的 GAN 损失
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))  # 判别伪造图片
        loss_G_GAN = self.criterionGAN(pred_fake, True)  # 生成器的 GAN 损失

        # 计算特征匹配损失
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:  # 如果启用特征匹配损失
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)  # 特征权重
            D_weights = 1.0 / self.opt.num_D  # 判别器权重
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # 计算 VGG 特征匹配损失
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:  # 如果启用 VGG 损失
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat

        # 返回损失值和生成图片（如果推理阶段需要）
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake), None if not infer else fake_image]

    def inference(self, label, inst, image=None):
        # 推理阶段的前向传播
        image = Variable(image) if image is not None else None  # 包装为 Variable
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)  # 编码输入

        # 生成伪造图片
        if self.use_features:  # 如果使用特征
            if self.opt.use_encoded_image:
                # 从真实图片中提取特征
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # 从预计算的特征簇中采样特征
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)  # 拼接特征
        else:
            input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():  # 禁用梯度计算
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        # 从预计算的特征簇中采样特征
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)  # 特征簇路径
        features_clustered = np.load(cluster_path, encoding='latin1').item()  # 加载特征簇

        # 随机从特征簇中采样
        inst_np = inst.cpu().numpy().astype(int)  # 转为 NumPy 数组
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])  # 初始化特征图
        for i in np.unique(inst_np):  # 遍历每个实例
            label = i if i < 1000 else i // 1000  # 提取标签
            if label in features_clustered:
                feat = features_clustered[label]  # 获取特征
                cluster_idx = np.random.randint(0, feat.shape[0])  # 随机选择簇索引

                idx = (inst == int(i)).nonzero()  # 获取实例的索引
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]  # 填充特征图
        if self.opt.data_type == 16:
            feat_map = feat_map.half()  # 如果是 16 位数据，转换为半精度
        return feat_map

    def encode_features(self, image, inst):
        # 编码特征
        image = Variable(image.cuda(), volatile=True)  # 转为 Variable
        feat_num = self.opt.feat_num  # 特征数
        h, w = inst.size()[2], inst.size()[3]  # 获取图像高度和宽度
        block_num = 32  # 块数
        feat_map = self.netE.forward(image, inst.cuda())  # 使用编码器提取特征
        inst_np = inst.cpu().numpy().astype(int)  # 转为 NumPy 数组
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):  # 遍历每个实例
            label = i if i < 1000 else i // 1000  # 提取标签
            idx = (inst == int(i)).nonzero()  # 获取实例索引
            num = idx.size()[0]  # 获取实例数量
            idx = idx[num // 2, :]  # 获取中间索引
            val = np.zeros((1, feat_num + 1))  # 初始化特征值
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]  # 填充特征值
            val[0, feat_num] = float(num) / (h * w // block_num)  # 计算实例占比
            feature[label] = np.append(feature[label], val, axis=0)  # 添加到特征字典
        return feature

    def get_edges(self, t):
        # 获取边缘图
        edge = torch.cuda.ByteTensor(t.size()).zero_()  # 初始化边缘图
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])  # 检测水平边缘
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])  # 检测水平边缘
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])  # 检测垂直边缘
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])  # 检测垂直边缘
        if self.opt.data_type == 16:
            return edge.half()  # 半精度
        else:
            return edge.float()  # 单精度

    def save(self, which_epoch):
        # 保存网络参数
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)  # 保存生成器
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)  # 保存判别器
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)  # 保存编码器

    def update_fixed_params(self):
        # 在固定全局生成器参数后，开始微调
        params = list(self.netG.parameters())  # 获取生成器参数
        if self.gen_features:
            params += list(self.netE.parameters())  # 包括编码器参数
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))  # 更新优化器
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        # 更新学习率
        lrd = self.opt.lr / self.opt.niter_decay  # 学习率衰减步长
        lr = self.old_lr - lrd  # 新学习率
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr  # 更新判别器学习率
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr  # 更新生成器学习率
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))  # 输出学习率更新信息
        self.old_lr = lr  # 保存当前学习率

# 推理模型，继承 Pix2PixHDModel
class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        # 推理阶段的前向传播
        label, inst = inp  # 获取输入
        return self.inference(label, inst)  # 调用推理方法


        
