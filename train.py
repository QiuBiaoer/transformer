############################################################################################################
# 1. 使用argparse类实现可以在训练的启动命令中指定超参数
# 2. 可以通过在启动命令中指定 --seed 随机数种子来固定网络的初始化方式，以达到结果可复现的效果
# 3. 使用了更高级的学习策略 cosine annealing：在训练的第一轮使用一个较小的lr（warm_up），从第二个epoch开始，随训练轮数逐渐减小lr。
# 4. 可以通过在启动命令中指定 --model 来选择使用的模型
# 5. 新加了weight-decay权重衰减项，防止过拟合
# 6. 新加了记录每个epoch的loss和acc的log文件及可用于tensorboard可视化的文件
# 7. 可以通过在启动命令中指定 --tensorboard 来进行tensorboard可视化, 默认不启用。
#    注意，使用tensorboad之前需要使用命令 "tensorboard --logdir=log_path"来启动，结果通过网页 http://localhost:6006/'查看可视化结果
############################################################################################################
# --model 可选的超参如下：
# alexnet    vgg    googlenet     resnet     densenet      mobilenet     shufflenet
# efficient    convnext     vision_transformer      swin_transformer
############################################################################################################
# 训练命令示例： # python train.py --model resnet18  --batch_size 64 --lr 0.001 --epoch 100 --classes_num 4
############################################################################################################

import argparse  # 用于解析命令行参数
import torch
import torch.optim as optim  # PyTorch中的优化器
from torch.utils.data import DataLoader  # PyTorch中用于加载数据的工具
from tqdm import tqdm  # 用于在循环中显示进度条
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
import torch.nn.functional as F  # PyTorch中的函数库
from torchvision import datasets  # PyTorch中的视觉数据集
import torchvision.transforms as transforms  # PyTorch中的数据变换操作
# from tensorboardX import SummaryWriter  # 用于创建TensorBoard日志的工具
import os  # Python中的操作系统相关功能
# from utils import AverageMeter, accuracy  # 自定义工具模块，用于计算模型的平均值和准确度
# from model import model_dict  # 自定义模型字典，包含了各种模型的定义
import numpy as np  # NumPy库，用于数值计算
import time  # Python中的时间相关功能
import random  # Python中的随机数生成器

parser = argparse.ArgumentParser() # 导入argparse模块，用于解析命令行参数
parser.add_argument("--model_names", type=str, default="vit") # 添加命令行参数，指定模型名称
parser.add_argument("--pre_trained", type=bool, default=False) #指定是否使用预训练模型，默认为False
parser.add_argument("--classes_num", type=int, default=4) # 指定类别数，默认为4
parser.add_argument("--dataset", type=str, default="dataset\COVID_19_Radiography_Dataset") # 指定数据集名称，默认为"new_COVID_19_Radiography_Dataset"
parser.add_argument("--batch_size", type=int, default=16) #   指定批量大小，默认为64
parser.add_argument("--epoch", type=int, default=20) #  指定训练轮次数，默认为20
parser.add_argument("--lr", type=float, default=0.01) #  指定学习率，默认为0.01
parser.add_argument("--momentum", type=float, default=0.9)  # 优化器的动量，默认为 0.9
parser.add_argument("--weight-decay", type=float, default=1e-4)  # 权重衰减（正则化项），默认为 5e-4
parser.add_argument("--seed", type=int, default=33) # 指定随机种子，默认为33
parser.add_argument("--gpu-id", type=int, default=0) # 指定GPU编号，默认为0
parser.add_argument("--print_freq", type=int, default=1)  # 打印训练信息的频率，默认为 1（每个轮次打印一次）
parser.add_argument("--exp_postfix", type=str, default="seed33")  # 实验结果文件夹的后缀，默认为 "seed33"
parser.add_argument("--txt_name", type=str, default="lr0.01_wd5e-4")  # 文本文件名称，默认为 "lr0.01_wd5e-4"

args = parser.parse_args()

def seed_torch(seed=74):
    # 设置随机数生成器的种子，确保实验的可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) # 设置环境变量 CUDA_VISIBLE_DEVICES，指定可见的 GPU 设备，仅在需要时使用特定的 GPU 设备进行训练

exp_name = args.exp_postfix  # 从命令行参数中获取实验名称后缀
exp_path = "./report/{}/{}/{}".format(args.dataset, args.model_names, exp_name)  # 创建实验结果文件夹的路径
os.makedirs(exp_path, exist_ok=True)

# dataloader
transform_train = transforms.Compose([transforms.RandomRotation(90), # 随机旋转图像
                                        transforms.Resize([256, 256]), # # 调整图像大小为 256x256 像素
                                        transforms.RandomCrop(224),  # 随机裁剪图像为 224x224 大小
                                        transforms.RandomHorizontalFlip(), # 随机水平翻转图像
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.3738, 0.3738, 0.3738), # # 对图像进行标准化
                                                            (0.3240, 0.3240, 0.3240))])
transform_test = transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.3738, 0.3738, 0.3738),
                                                            (0.3240, 0.3240, 0.3240))])
trainset = datasets.ImageFolder(root=os.path.join(r'dataset\COVID_19_Radiography_Dataset', 'train'),
                                transform=transform_train)
testset = datasets.ImageFolder(root=os.path.join(r'dataset\COVID_19_Radiography_Dataset', 'val'),
                                transform=transform_test)

# 创建训练数据加载器
train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4,
                                           # 后台工作线程数量，可以并行加载数据以提高效率
                                           shuffle=True, pin_memory=True)  # 如果可用，将数据加载到 GPU 内存中以提高训练速度
# 创建测试数据加载器
test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=4,
                                          shuffle=False, pin_memory=True)

# train
def train_one_epoch(model, optimizer, train_loader):
    model.train()
    acc_recorder = AverageMeter()  # 用于记录精度的工具
    loss_recorder = AverageMeter()  # 用于记录损失的工具

    for (inputs, targets) in tqdm(train_loader, desc="train"): # 遍历训练数据加载器 train_loader 中的每个批次数据，使用 tqdm 包装以显示进度条。
        # for i, (inputs, targets) in enumerate(train_loader):
        if torch.cuda.is_available():  # 如果当前设备支持 CUDA 加速，则将输入数据和目标数据送到 GPU 上进行计算，设置 non_blocking=True 可以使数据异步加载，提高效率。
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        out = model(inputs)
        loss = F.cross_entropy(out, targets)  # 计算损失（交叉熵损失）
        loss_recorder.update(loss.item(), n=inputs.size(0))  # 记录损失值 # 调用 update 方法，传入当前批次的损失值 loss.item() 和该批次的样本数量 inputs.size(0)。 # 这样做是为了根据样本数量加权计算损失的平均值，确保不同批次的损失贡献相等，而不受批次大小的影响。
        acc = accuracy(out, targets)[0]  # 计算精度
        acc_recorder.update(acc.item(), n=inputs.size(0))  # 记录精度值
        optimizer.zero_grad()  # 清零之前的梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

    losses = loss_recorder.avg  # 计算平均损失
    acces = acc_recorder.avg  # 计算平均精度

    return losses, acces  # 返回平均损失和平均精度

def evaluation(model, test_loader):
    # 将模型设置为评估模式，不会进行参数更新
    model.eval()
    acc_recorder = AverageMeter()  # 初始化两个计量器，用于记录准确度和损失
    loss_recorder = AverageMeter()

    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Evaluating"):
            # for img, label in test_loader:   # 迭代测试数据加载器中的每个批次
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            acc = accuracy(out, label)[0]  # 计算准确度和损失
            loss = F.cross_entropy(out, label) # 计算交叉熵损失
            acc_recorder.update(acc.item(), img.size(0))  # 更新准确率记录器，记录当前批次的准确率  img.size(0)表示批次中的样本数量
            loss_recorder.update(loss.item(), img.size(0))  # 更新损失记录器，记录当前批次的损失
    losses = loss_recorder.avg # 计算所有批次的平均损失
    acces = acc_recorder.avg # 计算所有批次的平均准确率
    return losses, acces # 返回平均损失和准确率

def train(model, optimizer, train_loader, test_loader, scheduler):
    since = time.time()  # 记录训练开始时间
    best_acc = -1  # 初始化最佳准确度为-1，以便跟踪最佳模型
    f = open(os.path.join(exp_path, "{}.txt".format(args.txt_name)), "w")  # 打开一个用于写入训练过程信息的文件

    for epoch in range(args.epoch):
        # 在训练集上执行一个周期的训练，并获取训练损失和准确度
        train_losses, train_acces = train_one_epoch(
            model, optimizer, train_loader
        )
        # 在测试集上评估模型性能，获取测试损失和准确度
        test_losses, test_acces = evaluation(model, test_loader)
        # 如果当前测试准确度高于历史最佳准确度，更新最佳准确度并保存模型参数
        if test_acces > best_acc:
            best_acc = test_acces
            state_dict = dict(epoch=epoch + 1, model=model.state_dict(), acc=test_acces)
            name = os.path.join(exp_path, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            torch.save(state_dict, name)

        scheduler.step()  # 更新学习率调度器

        tags = ['train_losses',  # 定义要记录的训练信息的标签
                'train_acces',
                'test_losses',
                'test_acces']
        tb_writer.add_scalar(tags[0], train_losses, epoch + 1)  # 将训练信息写入TensorBoard
        tb_writer.add_scalar(tags[1], train_acces, epoch + 1)
        tb_writer.add_scalar(tags[2], test_losses, epoch + 1)
        tb_writer.add_scalar(tags[3], test_acces, epoch + 1)

        # 打印训练过程信息，以及将信息写入文件
        if (epoch + 1) % args.print_freq == 0: #  print_freq指定为1 则每轮都打印
            msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f}  test loss{:.2f} acc:{:.2f}\n".format(
                epoch + 1,
                args.model_names,
                train_losses,
                train_acces,
                test_losses,
                test_acces,
            )
            print(msg)
            f.write(msg)
            f.flush()
    # 输出训练结束后的最佳准确度和总训练时间
    msg_best = "model:{} best acc:{:.2f}\n".format(args.model_names, best_acc)
    time_elapsed = "traninng time: {}".format(time.time() - since)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()

if __name__ == "__main__":
    tb_path = "runs/{}/{}/{}".format(args.dataset, args.model_names,  # 创建 TensorBoard 日志目录路径
                                     args.exp_postfix)
    tb_writer = SummaryWriter(log_dir=tb_path)
    lr = args.lr
    model = model_dict[args.model_names](num_classes=args.classes_num, pretrained=args.pre_trained)  # 根据命令行参数创建神经网络模型
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.SGD(  # 创建随机梯度下降 (SGD) 优化器
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)  # 创建余弦退火学习率调度器  自动调整lr

    train(model, optimizer, train_loader, test_loader, scheduler)  # 开始训练过程