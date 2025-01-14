def CreateDataLoader(opt):
    # 导入自定义数据加载器的类 CustomDatasetDataLoader
    from data.custom_dataset_data_loader import CustomDatasetDataLoader

    # 创建 CustomDatasetDataLoader 类的实例
    data_loader = CustomDatasetDataLoader()

    # 打印数据加载器的名称，用于确认和调试
    print(data_loader.name())

    # 调用数据加载器的初始化方法，并传入选项参数 opt
    # 这会设置数据加载器的内部配置（例如数据路径、图像预处理等）
    data_loader.initialize(opt)

    # 返回初始化好的数据加载器实例
    return data_loader