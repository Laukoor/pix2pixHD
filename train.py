import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions

# 计算最小公倍数（lcm），用于频率参数的处理
def lcm(a,b): 
    return abs(a * b)//fractions.gcd(a,b) if a and b else 0

# 导入必要的模块和方法
from options.train_options import TrainOptions  # 训练参数选项
from data.data_loader import CreateDataLoader  # 数据加载器
from models.models import create_model  # 创建模型
import util.util as util  # 工具模块
from util.visualizer import Visualizer  # 可视化模块

# 初始化训练选项并解析参数
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')  # 保存迭代状态的路径

# 如果继续训练，加载之前的迭代状态
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# 计算打印频率和批大小的最小公倍数，确保同步
opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:  # 如果启用调试模式，设置参数
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

# 创建数据加载器并加载数据
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)  # 获取数据集大小
print('#training images = %d' % dataset_size)

# 创建模型和可视化工具
model = create_model(opt)
visualizer = Visualizer(opt)

# 如果启用混合精度训练，使用 apex 库进行初始化
if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D  # 获取生成器和判别器优化器

# 计算总的训练步数
total_steps = (start_epoch-1) * dataset_size + epoch_iter

# 初始化显示、打印和保存的步数偏移
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

# 开始训练循环
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()  # 记录当前 epoch 开始时间
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size  # 如果不是起始 epoch，重置迭代次数
    
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()  # 记录当前迭代开始时间
        total_steps += opt.batchSize  # 更新总步数
        epoch_iter += opt.batchSize  # 更新当前 epoch 的步数

        # 是否需要保存伪造图片
        save_fake = total_steps % opt.display_freq == display_delta

        # 前向传播
        losses, generated = model(
            Variable(data['label']), 
            Variable(data['inst']),
            Variable(data['image']), 
            Variable(data['feat']), 
            infer=save_fake
        )

        # 汇总各设备上的损失
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))  # 将损失值与损失名称绑定

        # 计算生成器和判别器的最终损失
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

        # 更新生成器的权重
        optimizer_G.zero_grad()
        if opt.fp16:  # 如果启用混合精度
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_G.backward()  # 反向传播生成器损失
        optimizer_G.step()  # 更新生成器参数

        # 更新判别器的权重
        optimizer_D.zero_grad()
        if opt.fp16:  # 如果启用混合精度
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_D.backward()  # 反向传播判别器损失
        optimizer_D.step()  # 更新判别器参数

        # 显示结果和错误
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)  # 打印当前错误
            visualizer.plot_current_errors(errors, total_steps)  # 绘制错误图

        # 显示生成结果
        if save_fake:
            visuals = OrderedDict([
                ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                ('synthesized_image', util.tensor2im(generated.data[0])),
                ('real_image', util.tensor2im(data['image'][0]))
            ])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # 保存最新的模型
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:  # 如果当前 epoch 的迭代数超过数据集大小，跳出循环
            break
       
    # 一个 epoch 结束
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # 保存当前 epoch 的模型
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    # 在指定 epoch 后训练整个网络
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    # 在 niter 之后，线性衰减学习率
    if epoch > opt.niter:
        model.module.update_learning_rate()
