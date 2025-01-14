import torch  # 导入 PyTorch 库，用于深度学习操作
import torch.nn as nn  # 从 PyTorch 导入神经网络模块
import functools  # 导入 functools，支持高阶函数操作
from torch.autograd import Variable  # 从自动求导模块导入 Variable（兼容旧版代码）
import numpy as np  # 导入 NumPy，用于数值计算

###############################################################################
# Functions
###############################################################################
# 定义权重初始化函数
# 用于为网络的不同层初始化权重，帮助模型更快收敛
def weights_init(m):
    classname = m.__class__.__name__  # 获取模块的类名
    if classname.find('Conv') != -1:  # 如果模块是卷积层
        m.weight.data.normal_(0.0, 0.02)  # 将卷积层的权重初始化为均值 0，标准差 0.02 的正态分布
    elif classname.find('BatchNorm2d') != -1:  # 如果模块是 2D 批归一化层
        m.weight.data.normal_(1.0, 0.02)  # 将批归一化层的权重初始化为均值 1，标准差 0.02 的正态分布
        m.bias.data.fill_(0)  # 将偏置初始化为 0

# 获取归一化层的方法
# 根据指定的归一化类型，动态返回批归一化或实例归一化
# norm_type: "batch" 或 "instance"
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':  # 如果是批归一化
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)  # 返回带可学习参数的批归一化层
    elif norm_type == 'instance':  # 如果是实例归一化
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)  # 返回不带可学习参数的实例归一化层
    else:  # 如果指定的归一化类型未实现
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)  # 抛出异常
    return norm_layer  # 返回归一化层

# 定义生成器
# 根据指定参数创建全局生成器、局部增强生成器或编码器
def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)  # 根据 norm 参数选择归一化层
    if netG == 'global':  # 如果生成器类型是 "global"
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)  # 创建全局生成器
    elif netG == 'local':  # 如果生成器类型是 "local"
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)  # 创建局部增强生成器
    elif netG == 'encoder':  # 如果生成器类型是 "encoder"
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)  # 创建编码器
    else:  # 如果指定的生成器类型未实现
        raise('generator not implemented!')  # 抛出异常
    print(netG)  # 打印生成器结构
    if len(gpu_ids) > 0:  # 如果指定了 GPU
        assert(torch.cuda.is_available())  # 确保 GPU 可用
        netG.cuda(gpu_ids[0])  # 将生成器移动到指定的 GPU
    netG.apply(weights_init)  # 初始化生成器的权重
    return netG  # 返回生成器

# 定义判别器
# 根据指定参数创建多尺度判别器
def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)  # 根据 norm 参数选择归一化层
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)  # 创建多尺度判别器
    print(netD)  # 打印判别器结构
    if len(gpu_ids) > 0:  # 如果指定了 GPU
        assert(torch.cuda.is_available())  # 确保 GPU 可用
        netD.cuda(gpu_ids[0])  # 将判别器移动到指定的 GPU
    netD.apply(weights_init)  # 初始化判别器的权重
    return netD  # 返回判别器

# 打印网络结构
# 打印网络结构和参数总量，用于调试和检查网络规模
def print_network(net):
    if isinstance(net, list):  # 如果网络是列表
        net = net[0]  # 取列表中的第一个网络
    num_params = 0  # 初始化参数计数
    for param in net.parameters():  # 遍历网络的所有参数
        num_params += param.numel()  # 累加参数的数量
    print(net)  # 打印网络结构
    print('Total number of parameters: %d' % num_params)  # 打印参数总数

##############################################################################
# Losses
##############################################################################
# 定义 GAN 损失
# 用于生成器和判别器的训练，支持 LSGAN 和标准 GAN
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()  # 初始化 GANLoss 类
        self.real_label = target_real_label  # 定义真实标签值
        self.fake_label = target_fake_label  # 定义伪造标签值
        self.real_label_var = None  # 初始化真实标签变量
        self.fake_label_var = None  # 初始化伪造标签变量
        self.Tensor = tensor  # 设置张量类型
        if use_lsgan:  # 如果使用 LSGAN
            self.loss = nn.MSELoss()  # 使用均方误差损失
        else:  # 如果使用标准 GAN
            self.loss = nn.BCELoss()  # 使用二元交叉熵损失

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None  # 初始化目标张量
        if target_is_real:  # 如果目标是真实的
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))  # 检查是否需要创建新标签
            if create_label:  # 如果需要
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)  # 创建真实标签张量
                self.real_label_var = Variable(real_tensor, requires_grad=False)  # 包装为 Variable
            target_tensor = self.real_label_var  # 设置目标张量为真实标签
        else:  # 如果目标是伪造的
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))  # 检查是否需要创建新标签
            if create_label:  # 如果需要
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)  # 创建伪造标签张量
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)  # 包装为 Variable
            target_tensor = self.fake_label_var  # 设置目标张量为伪造标签
        return target_tensor  # 返回目标张量

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):  # 如果输入是嵌套列表
            loss = 0  # 初始化损失
            for input_i in input:  # 遍历每个输入
                pred = input_i[-1]  # 获取最后一层的预测值
                target_tensor = self.get_target_tensor(pred, target_is_real)  # 获取目标张量
                loss += self.loss(pred, target_tensor)  # 累加损失
            return loss  # 返回总损失
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)  # 获取目标张量
            return self.loss(input[-1], target_tensor)  # 计算并返回损失

# 定义 VGG 感知损失
# 用于生成器的感知损失，提升生成图像的感知质量
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()  # 初始化 VGGLoss 类
        self.vgg = Vgg19().cuda()  # 初始化预训练 VGG19 模型并移动到 GPU
        self.criterion = nn.L1Loss()  # 使用 L1 损失函数
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # 定义每层特征的权重

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)  # 提取输入和目标的 VGG 特征
        loss = 0  # 初始化损失值
        for i in range(len(x_vgg)):  # 遍历所有特征层
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())  # 计算加权特征差异
        return loss  # 返回感知损失

##############################################################################
# Generator
##############################################################################
# 定义局部增强生成器类
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()  # 调用父类的初始化方法
        self.n_local_enhancers = n_local_enhancers  # 设置局部增强器的数量

        ###### 全局生成器模型 #####           
        ngf_global = ngf * (2**n_local_enhancers)  # 全局生成器的通道数量
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model  # 创建全局生成器模型
        model_global = [model_global[i] for i in range(len(model_global)-3)]  # 移除全局生成器的最终卷积层
        self.model = nn.Sequential(*model_global)  # 使用 nn.Sequential 将其封装为模型

        ###### 局部增强层 #####
        for n in range(1, n_local_enhancers+1):
            ### 下采样层
            ngf_global = ngf * (2**(n_local_enhancers-n))  # 每层的通道数量
            model_downsample = [
                nn.ReflectionPad2d(3),  # 添加反射填充
                nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),  # 卷积层
                norm_layer(ngf_global), nn.ReLU(True),  # 批归一化和激活函数
                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),  # 降采样
                norm_layer(ngf_global * 2), nn.ReLU(True)  # 批归一化和激活函数
            ]

            ### 残差块
            model_upsample = []  # 初始化上采样模型
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]  # 添加残差块

            ### 上采样层
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),  # 反卷积
                norm_layer(ngf_global), nn.ReLU(True)  # 批归一化和激活函数
            ]      

            ### 最终卷积层
            if n == n_local_enhancers:                
                model_upsample += [
                    nn.ReflectionPad2d(3),  # 添加反射填充
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()  # 卷积层和 Tanh 激活
                ]                       

            # 将下采样和上采样模型设置为对象的属性
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)  # 平均池化用于降采样

    def forward(self, input): 
        ### 创建输入金字塔
        input_downsampled = [input]  # 初始化降采样列表
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))  # 逐层降采样

        ### 在最粗粒度级别计算输出
        output_prev = self.model(input_downsampled[-1])        

        ### 一层一层构建输出
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')  # 获取下采样模型
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')  # 获取上采样模型
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]  # 当前层的输入
            output_prev = model_upsample(model_downsample(input_i) + output_prev)  # 合并结果并进行上采样
        return output_prev  # 返回最终输出

# 定义全局生成器
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)  # 确保残差块数量合法
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)  # 定义激活函数

        model = [
            nn.ReflectionPad2d(3),  # 添加反射填充
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation  # 卷积层和激活
        ]
        
        ### 下采样层
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),  # 降采样
                norm_layer(ngf * mult * 2), activation  # 批归一化和激活
            ]

        ### 残差块
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### 上采样层         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ngf * mult / 2)), activation  # 反卷积、归一化和激活
            ]
        model += [
            nn.ReflectionPad2d(3),  # 添加反射填充
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()  # 最终卷积和 Tanh 激活
        ]        
        self.model = nn.Sequential(*model)  # 将模型打包为顺序结构
            
    def forward(self, input):
        return self.model(input)  # 前向传播，生成输出       
        
# 定义残差块
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)  # 构建卷积块

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0  # 初始化填充大小
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]  # 添加反射填充
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]  # 添加复制填充
        elif padding_type == 'zero':
            p = 1  # 使用零填充
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)  # 抛出异常

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),  # 卷积层
            norm_layer(dim),  # 批归一化
            activation  # 激活函数
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]  # 如果使用 Dropout，添加 Dropout 层

        p = 0  # 初始化填充大小
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]  # 添加反射填充
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]  # 添加复制填充
        elif padding_type == 'zero':
            p = 1  # 使用零填充
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)  # 抛出异常

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),  # 第二个卷积层
            norm_layer(dim)  # 批归一化
        ]

        return nn.Sequential(*conv_block)  # 返回顺序卷积块

    def forward(self, x):
        out = x + self.conv_block(x)  # 输入与卷积块的输出相加（残差连接）
        return out  # 返回输出


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()  # 调用父类的初始化方法
        self.output_nc = output_nc  # 保存输出通道数

        # 初始化模型的第一部分，包含反射填充、卷积层、归一化层和激活函数
        model = [
            nn.ReflectionPad2d(3),  # 添加 3 像素的反射填充
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),  # 7x7 卷积，无填充
            norm_layer(ngf),  # 批归一化
            nn.ReLU(True)  # ReLU 激活函数
        ]             

        ### 下采样部分
        for i in range(n_downsampling):  # 循环构建多层下采样
            mult = 2**i  # 通道数倍增
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),  # 3x3 卷积，步长为 2
                norm_layer(ngf * mult * 2),  # 批归一化
                nn.ReLU(True)  # ReLU 激活函数
            ]

        ### 上采样部分
        for i in range(n_downsampling):  # 循环构建多层上采样
            mult = 2**(n_downsampling - i)  # 通道数依次减半
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),  # 反卷积操作
                norm_layer(int(ngf * mult / 2)),  # 批归一化
                nn.ReLU(True)  # ReLU 激活函数
            ]        

        # 最后的反射填充和卷积操作，生成最终输出
        model += [
            nn.ReflectionPad2d(3),  # 添加 3 像素的反射填充
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),  # 7x7 卷积，无填充
            nn.Tanh()  # Tanh 激活函数，输出范围为 [-1, 1]
        ]
        self.model = nn.Sequential(*model)  # 将所有模块封装为顺序模型

    def forward(self, input, inst):
        outputs = self.model(input)  # 通过模型计算输出

        # 按实例进行平均池化
        outputs_mean = outputs.clone()  # 克隆输出用于存储平均池化结果
        inst_list = np.unique(inst.cpu().numpy().astype(int))  # 获取实例标签的唯一值
        for i in inst_list:  # 遍历每个实例标签
            for b in range(input.size()[0]):  # 遍历每个 batch 的输入
                indices = (inst[b:b+1] == int(i)).nonzero()  # 获取实例对应的索引
                for j in range(self.output_nc):  # 遍历输出通道
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]  # 获取实例的特征值
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)  # 计算特征均值并扩展维度
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat  # 用均值更新输出
        return outputs_mean  # 返回平均池化后的输出


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()  # 调用父类的初始化方法
        self.num_D = num_D  # 判别器的数量
        self.n_layers = n_layers  # 每个判别器的层数
        self.getIntermFeat = getIntermFeat  # 是否获取中间特征
     
        for i in range(num_D):  # 遍历每个判别器
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)  # 创建单尺度判别器
            if getIntermFeat:  # 如果需要中间特征
                for j in range(n_layers+2):  # 遍历每一层，包括输入和输出层
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))  # 保存每一层为属性
            else:
                setattr(self, 'layer'+str(i), netD.model)  # 只保存完整模型

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)  # 下采样层，减少输入尺寸

    def singleD_forward(self, model, input):
        if self.getIntermFeat:  # 如果需要中间特征
            result = [input]  # 初始化结果列表
            for i in range(len(model)):  # 遍历模型的每一层
                result.append(model[i](result[-1]))  # 将上一层的输出作为下一层的输入
            return result[1:]  # 返回所有中间层的输出
        else:
            return [model(input)]  # 仅返回最终输出

    def forward(self, input):        
        num_D = self.num_D  # 获取判别器的数量
        result = []  # 初始化结果列表
        input_downsampled = input  # 初始化输入
        for i in range(num_D):  # 遍历每个判别器
            if self.getIntermFeat:  # 如果需要中间特征
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]  # 获取每一层的模型
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))  # 获取完整模型
            result.append(self.singleD_forward(model, input_downsampled))  # 将结果添加到结果列表中
            if i != (num_D-1):  # 如果不是最后一个判别器
                input_downsampled = self.downsample(input_downsampled)  # 对输入进行下采样
        return result  # 返回所有判别器的输出
        
# 定义具有指定参数的 PatchGAN 判别器
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()  # 调用父类初始化方法
        self.getIntermFeat = getIntermFeat  # 是否获取中间特征
        self.n_layers = n_layers  # 判别器的层数

        kw = 4  # 卷积核大小为 4
        padw = int(np.ceil((kw-1.0)/2))  # 计算填充大小以保持特征图尺寸
        # 初始化第一层，输入通道为 input_nc，输出通道为 ndf
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf  # 当前通道数
        for n in range(1, n_layers):  # 添加中间层
            nf_prev = nf  # 保存上一层的通道数
            nf = min(nf * 2, 512)  # 通道数翻倍，最多 512
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),  # 卷积层
                norm_layer(nf), nn.LeakyReLU(0.2, True)  # 批归一化和 LeakyReLU 激活
            ]]

        nf_prev = nf  # 最后一层的输入通道数
        nf = min(nf * 2, 512)  # 最后一层的输出通道数
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),  # 卷积层
            norm_layer(nf),  # 批归一化
            nn.LeakyReLU(0.2, True)  # LeakyReLU 激活
        ]]

        # 输出层，输出单通道特征图（1 表示真实，0 表示伪造）
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:  # 如果指定使用 Sigmoid 激活
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:  # 如果需要获取中间特征
            for n in range(len(sequence)):  # 遍历每一层
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))  # 将每层保存为属性
        else:
            sequence_stream = []  # 顺序模型
            for n in range(len(sequence)):
                sequence_stream += sequence[n]  # 合并所有层
            self.model = nn.Sequential(*sequence_stream)  # 创建顺序模型

    def forward(self, input):
        if self.getIntermFeat:  # 如果需要获取中间特征
            res = [input]  # 初始化结果列表
            for n in range(self.n_layers+2):  # 遍历每一层，包括输入和输出层
                model = getattr(self, 'model'+str(n))  # 获取每层模型
                res.append(model(res[-1]))  # 逐层向前传播
            return res[1:]  # 返回中间特征
        else:
            return self.model(input)  # 返回最终输出

# 定义 VGG19 模型类
from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()  # 调用父类初始化方法
        vgg_pretrained_features = models.vgg19(pretrained=True).features  # 加载预训练的 VGG19 特征提取部分

        # 按照 VGG19 的层结构切分为多个部分
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):  # 第一部分包含前两层
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):  # 第二部分包含第 3 到 7 层
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):  # 第三部分包含第 8 到 12 层
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):  # 第四部分包含第 13 到 21 层
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):  # 第五部分包含第 22 到 30 层
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:  # 如果不需要更新权重
            for param in self.parameters():
                param.requires_grad = False  # 冻结所有参数

    def forward(self, X):
        # 按顺序通过每个部分，提取多层特征
        h_relu1 = self.slice1(X)  # 第一部分
        h_relu2 = self.slice2(h_relu1)  # 第二部分
        h_relu3 = self.slice3(h_relu2)  # 第三部分
        h_relu4 = self.slice4(h_relu3)  # 第四部分
        h_relu5 = self.slice5(h_relu4)  # 第五部分
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]  # 收集所有特征
        return out  # 返回特征列表
