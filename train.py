from typing import Tuple
from numpy.core.arrayprint import printoptions
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import logging
from torch.nn import functional as F
from torch import Tensor
import tqdm
import numpy as np
import os
import argparse

from dataset.myDataset import MyDataset
from model.elimator import Elimator

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--warning_rate', type=float, default=5e-5, help='lr for first epoch')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--model_dir', type=str, help='if continue train')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save_path', type=str, default='output/')
parser.add_argument('--save_epoch', type=int, default=4, help='save every n epochs')
parser.add_argument('--data_path', type=str, default="/home/Data/CarData/" ,help='root of data, consist train and test')
parser.add_argument('--device', type=str, default="cuda:0", help='chose device to train(only support single)')
parser.add_argument('--log_file', type=str, default='train.log', help='path to save train log')
parser.add_argument('--model_name', type=str, default='model', help='save checkpoint name')

class Trainer:
    def __init__(self,
                 batch_size,
                 lr,  # 学习率
                 data_path,
                 device='cuda:0',
                 log_file='train.log',
                 model_name='model'
                 ):
        self.batch_size = batch_size
        self.lr = lr

        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available()
        self.device = torch.device(device if cuda_condition else "cpu")
        # 将模型发送到计算设备(GPU或CPU)
        self.model = Elimator()
        self.model.to(self.device)

        # 声明训练数据集
        train_dataset = MyDataset(data_path + '/train')
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=0,
                                           shuffle=True
                                           )
        # 声明测试数据集
        test_dataset = MyDataset(data_path + '/test')
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=0,
                                          shuffle=True
                                          )
        

        self.init_optimizer(lr=self.lr)
        # self.criterion = self.init_criterion()

        self.log_file = log_file
        self.logger = self.init_logger(self.log_file)
        self.model_name = model_name

    def init_optimizer(self, lr):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.lr = lr

    def adjust_optimizer(self, lr):
        # 用指定的学习率调整化优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def criterion(self, img: Tensor, outputs: Tensor, gt: Tensor):
        # 构建损失函数
        

        return nn.CrossEntropyLoss()

    def init_logger(self, file_name):
        # 1.显示创建
        logging.basicConfig(filename=file_name, format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        # 2.定义logger,设定setLevel，FileHandler，setFormatter
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        logger.addHandler(console)
        return logger

    def load_model(self, model, model_dir="output"):
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded!".format(model_dir))

    def train(self, epoch):
        # 一个epoch的训练
        self.model.train()
        self.iteration(epoch, self.train_dataloader, train=True)

    def test(self, epoch):
        # 一个epoch的测试
        self.model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, data_loader, train=True):
        # 进度条显示
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # 所有的指标
        avg_loss = Averager()
        acc1 = Averager()
        acc5 = Averager()
        
        for i, data in data_iter:
            image = data['image'].float().to(self.device)
            label = data['label'].to(self.device)
            output = self.model(image)

            loss = self.criterion(output, label)

            # 指标计算
            acc = self.topkAcc(output, label,topk=(1, 5))
            acc1.update(acc[0])
            acc5.update(acc[1])
            avg_loss.update(loss.item())

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if i % 50 == 0:
                log_dic = {}
                if train:
                    log_dic = {
                        "epoch": epoch, "state": str_code,
                        "train_loss": avg_loss(), "train_acc1": acc1(),
                        "train_acc5": acc5()
                    }

                else:
                    log_dic = {
                        "epoch": epoch, "state": str_code,
                        "test_acc1": acc1(), "test_acc5": acc5()
                    }
                print(log_dic)

        log_dic = {"epoch": epoch, "state": str_code, "acc1": acc1(), "acc5": acc5()}
        if train:
            log_dic['train_loss'] = avg_loss()
            log_dic['lr'] = self.lr
        self.logger.info(log_dic)

        return acc1()

    def topkAcc(self, preds:torch.Tensor, labels:torch.Tensor, topk=(1,)):
        assert preds.shape[0] == labels.shape[0]
        batch_size = preds.shape[0]
        result = []
        for k in topk:
            cnt = 0
            values, indexs = preds.topk(k)
            for i in range(batch_size):
                if labels[i] in indexs[i]:
                    cnt += 1
            result.append(cnt/batch_size)
        return result
        

    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, epoch, state_dict_dir="output/"):
        """存储当前模型参数"""
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + self.model_name + ".{}.pth".format(str(epoch))
        model.to("cpu")
        model.eval()
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print("{} saved!".format(save_path))
        model.to(self.device)

class Averager():
    def __init__(self) -> None:
        self.sum = 0
        self.n = 0
        self.avg = 0
    
    def update(self, value, n=1):
        self.sum += value
        self.n += n
        self.avg = self.sum / self.n
    
    def __call__(self) -> float:
        return self.avg


if __name__ == '__main__':
    args = parser.parse_args()
    start_epoch = 0
    trainer = Trainer(args.batch_size, args.warning_rate, data_path=args.data_path, device=args.device, 
                    log_file=args.log_file, model_name=args.model_name)

    all_acc = []
    threshold = 999
    patient = 10
    best_loss = 999999999
    
    lr = args.warning_rate if start_epoch == 0 else args.lr
    if args.model_dir is not None:
        trainer.load_model(args.model_dir)
    for epoch in range(start_epoch, start_epoch + args.epoch):
        print("train with learning rate {}".format(str(trainer.lr)))
        # 训练一个epoch
        trainer.train(epoch)
        if epoch == start_epoch + args.warning_epoch - 1:
            lr = args.lr
            trainer.adjust_optimizer(lr)
        
        acc = trainer.test(epoch)
        if epoch % args.save_epoch == 0:
            # 保存当前epoch模型参数
            trainer.save_state_dict(epoch, args.save_path)

        if acc > 0.98:
            trainer.logger.info('train finish, epoch=%d, acc=%d' % (epoch, acc))
            break
        all_acc.append(acc)
        best_acc = max(all_acc)
        if all_acc[-1] < best_acc:
            threshold += 1
            lr *= 0.9
            trainer.adjust_optimizer(lr=lr)
        else:
            # 如果
            threshold = 0
            lr *= 1.1
            trainer.adjust_optimizer(lr=lr)

        if threshold >= patient:
            print("epoch {} has the lowest loss".format(start_epoch + np.argmax(np.array(all_acc))))
            print("early stop!")
            break
