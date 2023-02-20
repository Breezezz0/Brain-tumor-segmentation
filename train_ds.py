

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


os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark


net = torch.nn.DataParallel(net).cuda()
net.train()


train_ds = Dataset(image_path[:444], label_path[:444])
valid_ds = Dataset(image_path[444:], label_path[444:540])


train_dl = DataLoader(train_ds, para.batch_size, True,
                      num_workers=para.num_workers, pin_memory=para.pin_memory)
valid_dl = DataLoader(valid_ds, para.batch_size, True,
                      num_workers=para.num_workers, pin_memory=para.pin_memory)

loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(
), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
loss_func = loss_func_list[5]


opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)


lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)


alpha = para.alpha


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

    if epoch % 5 == 0 and epoch != 0:

        oss
        torch.save(net.state_dict(
        ), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))

    if epoch % 5 == 0 and epoch != 0:
        alpha *= 0.8
