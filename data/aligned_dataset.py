import os.path  # 用于处理文件路径
from data.base_dataset import BaseDataset, get_params, get_transform, normalize  # 导入基础数据集和相关函数
from data.image_folder import make_dataset  # 导入用于生成数据集路径的函数
from PIL import Image  # 图像处理库

# 定义 AlignedDataset 类，继承自 BaseDataset
class AlignedDataset(BaseDataset):
    # 初始化数据集
    def initialize(self, opt):
        self.opt = opt  # 保存选项参数
        self.root = opt.dataroot  # 数据集根目录

        ### input A (标签图)
        # 根据选项决定标签图文件夹的后缀
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        # 拼接路径，确定标签图文件夹路径
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        # 获取标签图的所有文件路径，并排序
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (真实图像)
        if opt.isTrain or opt.use_encoded_image:  # 如果是训练模式或使用编码图像
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            # 拼接路径，确定真实图像文件夹路径
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            # 获取真实图像的所有文件路径，并排序
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps (实例图)
        if not opt.no_instance:  # 如果需要实例图
            # 拼接路径，确定实例图文件夹路径
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            # 获取实例图的所有文件路径，并排序
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features (预加载特征)
        if opt.load_features:  # 如果需要加载预计算的特征
            # 拼接路径，确定特征文件夹路径
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            # 获取特征的所有文件路径，并排序
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        # 保存数据集大小
        self.dataset_size = len(self.A_paths)

    # 获取单个样本数据
    def __getitem__(self, index):        
        ### input A (标签图)
        A_path = self.A_paths[index]  # 获取当前索引的标签图路径
        A = Image.open(A_path)  # 打开标签图
        params = get_params(self.opt, A.size)  # 获取图像变换参数
        if self.opt.label_nc == 0:  # 如果标签是 RGB 格式
            transform_A = get_transform(self.opt, params)  # 获取标准变换
            A_tensor = transform_A(A.convert('RGB'))  # 转换为张量
        else:  # 如果标签是单通道（语义分割）
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)  # 最近邻插值
            A_tensor = transform_A(A) * 255.0  # 放大值范围到 [0, 255]

        # 初始化其他输入为 0
        B_tensor = inst_tensor = feat_tensor = 0

        ### input B (真实图像)
        if self.opt.isTrain or self.opt.use_encoded_image:  # 如果是训练模式或使用编码图像
            B_path = self.B_paths[index]  # 获取当前索引的真实图像路径
            B = Image.open(B_path).convert('RGB')  # 打开并转换为 RGB 图像
            transform_B = get_transform(self.opt, params)  # 获取标准变换
            B_tensor = transform_B(B)  # 转换为张量

        ### 如果使用实例图
        if not self.opt.no_instance:  # 如果需要实例图
            inst_path = self.inst_paths[index]  # 获取当前索引的实例图路径
            inst = Image.open(inst_path)  # 打开实例图
            inst_tensor = transform_A(inst)  # 转换为张量

            if self.opt.load_features:  # 如果需要加载预计算特征
                feat_path = self.feat_paths[index]  # 获取当前索引的特征路径
                feat = Image.open(feat_path).convert('RGB')  # 打开并转换为 RGB 图像
                norm = normalize()  # 获取标准归一化
                feat_tensor = norm(transform_A(feat))  # 转换并归一化特征张量

        # 将所有输入封装为字典
        input_dict = {
            'label': A_tensor,  # 标签图张量
            'inst': inst_tensor,  # 实例图张量
            'image': B_tensor,  # 真实图像张量
            'feat': feat_tensor,  # 特征张量
            'path': A_path  # 标签图路径
        }

        return input_dict  # 返回字典

    # 获取数据集长度
    def __len__(self):
        # 返回数据集大小（受 batchSize 对齐的限制）
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    # 返回数据集名称
    def name(self):
        return 'AlignedDataset'
