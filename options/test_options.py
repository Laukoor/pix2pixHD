from .base_options import BaseOptions  # 导入 BaseOptions 类

# 定义测试选项类，继承自 BaseOptions
class TestOptions(BaseOptions):
    def initialize(self):  
        BaseOptions.initialize(self)  # 调用父类的 initialize 方法，加载基础选项

        # 添加新的测试选项参数
        self.parser.add_argument('--ntest', type=int, default=float("inf"), 
                                 help='# of test examples.')  # 测试的样本数量，默认为无穷大
        self.parser.add_argument('--results_dir', type=str, default='./results/', 
                                 help='saves results here.')  # 测试结果的保存目录
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, 
                                 help='aspect ratio of result images')  # 结果图像的宽高比
        self.parser.add_argument('--phase', type=str, default='test', 
                                 help='train, val, test, etc')  # 测试阶段的名称，默认为 "test"
        self.parser.add_argument('--which_epoch', type=str, default='latest', 
                                 help='which epoch to load? set to latest to use latest cached model')  
                                 # 加载模型的 epoch，默认为 "latest"
        self.parser.add_argument('--how_many', type=int, default=50, 
                                 help='how many test images to run')  # 测试图像数量，默认为 50
        self.parser.add_argument('--cluster_path', type=str, 
                                 default='features_clustered_010.npy', 
                                 help='the path for clustered results of encoded features')  
                                 # 预训练特征的聚类文件路径
        self.parser.add_argument('--use_encoded_image', action='store_true', 
                                 help='if specified, encode the real image to get the feature map')  
                                 # 是否对真实图像编码以获取特征图
        self.parser.add_argument("--export_onnx", type=str, 
                                 help="export ONNX model to a given file")  
                                 # 导出 ONNX 格式模型的路径
        self.parser.add_argument("--engine", type=str, 
                                 help="run serialized TRT engine")  
                                 # 运行序列化的 TensorRT 引擎
        self.parser.add_argument("--onnx", type=str, 
                                 help="run ONNX model via TRT")  
                                 # 使用 TensorRT 运行 ONNX 模型

        # 设置为测试模式
        self.isTrain = False

