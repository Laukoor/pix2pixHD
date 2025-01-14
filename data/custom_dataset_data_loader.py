import torch.utils.data  # 导入 PyTorch 数据加载工具
from data.base_data_loader import BaseDataLoader  # 导入基础数据加载器类

# 定义用于创建数据集的函数
def CreateDataset(opt):
    dataset = None  # 初始化数据集变量
    from data.aligned_dataset import AlignedDataset  # 动态导入 AlignedDataset 类
    dataset = AlignedDataset()  # 创建 AlignedDataset 的实例

    # 打印数据集名称，便于调试和确认
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)  # 调用数据集的初始化方法，将选项 opt 应用于数据集
    return dataset  # 返回创建并初始化的数据集实例

# 自定义数据加载器类，继承自 BaseDataLoader
class CustomDatasetDataLoader(BaseDataLoader):
    # 定义类名称的返回方法
    def name(self):
        return 'CustomDatasetDataLoader'

    # 初始化自定义数据加载器
    def initialize(self, opt):
        # 调用父类 BaseDataLoader 的初始化方法
        BaseDataLoader.initialize(self, opt)
        
        # 创建数据集实例
        self.dataset = CreateDataset(opt)
        
        # 创建 PyTorch 数据加载器
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,  # 使用之前创建的数据集
            batch_size=opt.batchSize,  # 每个批次的大小
            shuffle=not opt.serial_batches,  # 是否随机打乱数据，默认打乱
            num_workers=int(opt.nThreads))  # 工作线程数，用于并行加载数据

    # 返回数据加载器实例，供外部使用
    def load_data(self):
        return self.dataloader

    # 返回数据集的长度（受最大数据集大小限制）
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
