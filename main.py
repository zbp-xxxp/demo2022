import paddle
print("Paddle Vision is ", paddle.__version__)

from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')

# 模型网络结构搭建
network = paddle.nn.Sequential(
    paddle.nn.Flatten(),           # 拉平，将 (28, 28) => (784)
    paddle.nn.Linear(784, 512),    # 隐层：线性变换层
    paddle.nn.ReLU(),              # 激活函数
    paddle.nn.Linear(512, 10)      # 输出层
)

model = paddle.Model(network )   # 用Model封装模型
# 模型可视化
model.summary((1, 28, 28))
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )

# 训练模型
model.fit(train_dataset,
        epochs=1,
        batch_size=1,
        verbose=1
        )

# 保存用于后续继续调优训练的模型
model.save('finetuning/mnist')
# 保存用于后续推理部署的模型
model.save('infer/mnist', training=False)

model.evaluate(test_dataset, batch_size=1, verbose=1)