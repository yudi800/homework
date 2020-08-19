import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from Dataset import gettraindataloder,gettestdataloder
from torch.autograd import Variable


class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        dropout = 0.5
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            # 16
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.classifier(out)
        return out


# 使用预训练模型


if __name__ == "__main__":
    # 使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 数据预处理
    # 图像预处理和增强
    lr = 0.001
    batch_size = 8
    epochs = 30
    trainLoder,trainLoder1 = gettraindataloder(batch_size)
    testLoder,testLoder1 = gettraindataloder(batch_size)
    # x = torch.ones((1, 6, 32, 32), dtype=torch.float32).to(device)
    netw = MyNet().to(device)
    # 预热
    # for i in range(5):
    #     y = netw(x)
    # # 多次推理取平均值
    # start = time.time()
    # for i in range(100):
    #     y = netw(x)
    # end = time.time()
    # print('Latency(ms):', (end - start) / 100 * 1000)


    # 模型定义 MyNet
    net = MyNet().to(device)

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.Adam(net.parameters(), lr=lr,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    for epoch in range(0, epochs):
        # 开始训练
        net.train()
        # 总损失
        sum_loss = 0.0
        # 准确率
        accuracy = 0.0
        total = 0.0
        for step, (img1, img2) in enumerate(zip(trainLoder,trainLoder1)):
            # 准备数据
             # 数据大小
            if step%100==0:
                print(step)
            inputs = torch.cat([img1[0],img2[0]],dim=1)  # 取出数据
            labels = img1[1]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
            inputs, labels = Variable(inputs), Variable(labels)
            # forward + backward + optimize
            outputs = net(inputs)  # 前向传播求出预测值
            loss = criterion(outputs, labels)  # 求loss
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 更新参数

            # 每一个batch输出对应的损失loss和准确率accuracy
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            accuracy += predicted.eq(labels.data).cpu().sum()  # 预测值和真实值进行比较，将数据放到cpu上并且求和
            # print(epoch,step,img1[1])
        with torch.no_grad():
            accuracy2 = 0
            total1 = 0.0
            totalstep=0.0
            for step, (img1, img2) in enumerate(zip(testLoder,testLoder1)):
                # 开始测试
                net.eval()

                images, labels = torch.cat([img1[0],img2[0]],dim=1),img1[1]
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引(得分高的那一类)
                total1 += labels.size(0)
                accuracy2 += (predicted == labels).sum()
                # print(epoch, step, img1[1])
                totalstep=step
            # 输出测试准确率
            # print(total,total1)
            if total1==0.0:
                total1=1
            if total==0.0:
                total=1
            if step==0.0:
                step=1
            # print('[epoch:%d] Loss: %.03f | Acc: %.3f%% Testacc: %.3f%%'
            #       % (epoch + 1, sum_loss / (totalstep + 1), 100. * accuracy / total, 100. * accuracy2 / total1))









