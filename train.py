from numpy.core.arrayprint import printoptions
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import logging
from torch.nn import functional as F
import tqdm
import numpy as np
import os
import argparse

from dataset.myDataset import MyDataset
from model.Inception_resnetv2 import Inception_ResNetv2

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--warning_rate', type=float, default=5e-5, help='lr for first epoch')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--model_dir', type=str, help='if continue train')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save_path', type=str, default='output/')
parser.add_argument('--save_epoch', type=int, default=4, help='save every n epochs')
parser.add_argument('--data_path', type=str, default="/home/Data/CarData/" ,help='root of data, consist train and test')

class Trainer:
    def __init__(self,
                 batch_size,
                 lr,  # 学习率
                 data_path,
                 classes=242,
                 with_cuda=True,  # 是否使用GPU, 如未找到GPU, 则自动切换CPU
                 ):
        self.batch_size = batch_size
        self.lr = lr
        self.classes = classes

        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # 将模型发送到计算设备(GPU或CPU)
        self.model = Inception_ResNetv2(classes=classes)
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
        self.criterion = self.init_criterion()
        self.logger = self.init_logger()

    def init_optimizer(self, lr):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.lr = lr

    def init_criterion(self):
        # 构建损失函数
        return nn.CrossEntropyLoss()

    def init_logger(self):
        # 1.显示创建
        logging.basicConfig(filename='train.log', format='%(asctime)s - %(levelname)s - %(message)s',
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

    def save_state_dict(self, model, epoch, state_dict_dir="output", file_name="car.model"):
        """存储当前模型参数"""
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + "/" + file_name + ".epoch.{}".format(str(epoch))
        model.to("cpu")
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
    trainer = Trainer(args.batch_size, args.warning_rate, data_path=args.data_path)

    all_acc = []
    threshold = 999
    patient = 10
    best_loss = 999999999
    if args.model_dir is not None:
        trainer.load_model(trainer.model, args.model_dir)
    for epoch in range(start_epoch, start_epoch + args.epoch):
        print("train with learning rate {}".format(str(trainer.lr)))
        # 训练一个epoch
        trainer.train(epoch)
        if epoch == start_epoch:
            trainer.init_optimizer(args.lr)
        if epoch % args.save_epoch == 0:
            # 保存当前epoch模型参数
            trainer.save_state_dict(trainer.model, epoch, args.save_path)

        acc = trainer.test(epoch)

        if acc > 0.98:
            trainer.logger.info('train finish, epoch=%d, acc=%d' % (epoch, acc))
            break
        all_acc.append(acc)
        best_acc = max(all_acc)
        if all_acc[-1] < best_acc:
            threshold += 1
            args.lr *= 0.8
            trainer.init_optimizer(lr=args.lr)
        else:
            # 如果
            threshold = 0

        if threshold >= patient:
            print("epoch {} has the lowest loss".format(start_epoch + np.argmax(np.array(all_acc))))
            print("early stop!")
            break
