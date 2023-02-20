"""

训练脚本
"""

import os
from time import time

import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset_3d_latest import HecktorDataset_one_patient as Dataset

from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss

from net.ResUNet import net

import parameter as para

image_dir = r"/home/u109001022/HECKTOR/hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022/imagesTr/"
label_dir = r"/home/u109001022/HECKTOR/hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022/labelsTr/"
image_path = []
label_path = []
for filename in os.listdir(image_dir):
    if 'CT' in filename:
        fullpath = os.path.join(image_dir, filename)
        if os.path.isfile(fullpath):
            image_path.append(fullpath)
for filename in os.listdir(label_dir):
    fullpath = os.path.join(label_dir, filename)
    if os.path.isfile(fullpath):
        label_path.append(fullpath)

_exp_name = 'Train_3d'


# 设置显卡相关
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# 定义网络
net = torch.nn.DataParallel(net).cuda()
net.train()

# 定义Dateset
train_ds = Dataset(image_path[:444], label_path[:444])
valid_ds = Dataset(image_path[444:], label_path[444:540])

# 定义数据加载
train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
valid_dl = DataLoader(valid_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)
# 挑选损失函数
loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
loss_func = loss_func_list[5]

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# 深度监督衰减系数
alpha = para.alpha

# 训练网络
start = time()
for epoch in tqdm(range(para.Epoch)):

    lr_decay.step()

    mean_loss = []
    net.train()
    for step, (ct, seg) in enumerate(train_dl):
        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        loss = (loss1 + loss2 + loss3) * alpha + loss4

        mean_loss.append(loss4.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5 == 0:
            with open(f"./{_exp_name}_train_log_3d.txt", "a") as f:
                  f.write(
                      'epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min\n'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))
    net.eval()
    for step, (ct, seg) in enumerate(valid_dl):
        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        loss = (loss1 + loss2 + loss3) * alpha + loss4

        mean_loss.append(loss4.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5 == 0:
            with open(f"./{_exp_name}_train_log_valid_3d.txt", "a") as f:
                  f.write(
                      'epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min\n'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))
            print('---------------------eval------------------------------')
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))
    mean_loss = sum(mean_loss) / len(mean_loss)

    # 保存模型
    if epoch % 5 == 0 and epoch != 0:

        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))

    # 对深度监督系数进行衰减
    if epoch % 5 == 0 and epoch != 0:
        alpha *= 0.8

# 深度监督的系数变化
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000
