import os
import sys
import torch
import tarfile
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dcase import DCaseDataset
from models.cnns import get_net, LogMelSpectrogram

import torch.nn.functional as F


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


class KnowledgeDistillation(nn.Module):
    def __init__(self, teacher=None, temperature=1.0, kd_loss_type='kl'):
        super(KnowledgeDistillation, self).__init__()
        self.kd_loss_type = kd_loss_type
        self.teacher = teacher.eval()
        self.T = temperature
        self.kd_loss_type = kd_loss_type
        self.kl_loss = torch.nn.KLDivLoss()
        self.l2_loss = torch.nn.MSELoss()

    def kd_logit(self, student_out, teacher_out):
        soft_teacher_pred = F.softmax(teacher_out / self.T)
        log_soft_pred = F.log_softmax(student_out / self.T)
        kd_loss = self.kl_loss(log_soft_pred, soft_teacher_pred)
        return kd_loss

    def feat_reg(self, student_out, teacher_out):
        kd_loss = 0
        for student_feat, teacher_feat in zip(student_out, teacher_out):
            kd_loss += self.l2_loss(student_feat, teacher_feat)
        return kd_loss

    def attention_transfer(self, student_out, teacher_out):
        kd_loss = 0
        for student_feat, teacher_feat in zip(student_out, teacher_out):
            kd_loss += at_loss(student_feat, teacher_feat)
        return kd_loss

    def forward(self, student_out, input):
        with torch.no_grad():
            teacher_out = self.teacher(input, out_feat=True)
        # breakpoint()
        kd_loss = [self.kd_logit(student_out[-1], teacher_out[-1])]
        if self.kd_loss_type == 'feat_reg':
            kd_loss.append(self.feat_reg(student_out[:-1], teacher_out[:-1]))
        elif self.kd_loss_type == 'at':  # Attention transfer
            kd_loss.append(self.attention_transfer(student_out[:-1], teacher_out[:-1]))
        return kd_loss


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())

    return res


class CELoss:

    def __init__(self, nb_classes):
        self.nb_classes = nb_classes

    def __call__(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        if target.ndim == 1:  # Assume target are indices
            target = F.one_hot(target, num_classes=self.nb_classes)
        return - (pred * target).sum(1).mean()


class MixUp:

    def __init__(self, alpha, nb_classes):
        self.nb_classes = nb_classes
        self.beta = torch.distributions.beta.Beta(alpha, alpha)

    @staticmethod
    def mix(x, mix, ind):
        return x * mix + x[ind] * (1 - mix)

    def __call__(self, input, target):
        # Transform to hot hot vector
        target = F.one_hot(target, num_classes=self.nb_classes)
        # Mix signals
        ind = torch.randperm(input.shape[0])
        mix = self.beta.sample()
        input = MixUp.mix(input, mix, ind)
        target = MixUp.mix(target, mix, ind)

        return input, target


class TrainingManager:

    def __init__(
            self,
            net,
            spectrogram,
            loader_train,
            loader_test,
            optim,
            scheduler,
            max_epoch,
            path,
            teacher_net,
            kd_loss_type,
            kd_alpha,
            temperature,
            kd_beta,
            mixup_alpha,
            reload_ckpt=False
    ):

        # Optim. methods
        self.optim = optim
        self.scheduler = scheduler

        # Dataloaders
        self.max_epoch = max_epoch
        self.loader_train = loader_train
        self.loader_test = loader_test

        # Networks
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.dev)
        self.spectrogram = spectrogram.to(self.dev).eval()

        # Loss
        # self.CEloss = nn.CrossEntropyLoss().to(self.dev)
        # Mixup and loss
        self.CEloss = CELoss(nb_classes=10)
        self.mixup = MixUp(alpha=mixup_alpha, nb_classes=10)

        # Distillation
        self.kd_loss_type = kd_loss_type
        self.kd_alpha = kd_alpha
        self.T = temperature
        self.beta = kd_beta
        if teacher_net:
            self.teacher_net = teacher_net.to(self.dev).eval()
            self.Distill = KnowledgeDistillation(teacher=teacher_net, temperature=temperature,
                                                 kd_loss_type=kd_loss_type).to(self.dev)

        # Checkpoints
        self.path_to_ckpt = path + '/ckpt.pth'
        if reload_ckpt:
            self.load_state()
        else:
            self.current_epoch = 0

        # Monitoring
        self.writer = SummaryWriter(
            path + '/tensorboard/', purge_step=self.current_epoch + 1
        )

    def print_log(self, running_loss, nb_it, acc1, acc5, nb_instances):
        log = '\nEpoch: {0:d} :'.format(self.current_epoch) + \
              ' loss = {0:.3f}'.format(running_loss / (nb_it + 1)) + \
              ' - acc1 = {0:.3f}'.format(100 * acc1 / nb_instances) + \
              ' - acc5 = {0:.3f}'.format(100 * acc5 / nb_instances)
        print(log)

    def one_epoch(self, training):

        # Train or eval mode
        if training:
            net = self.net.train()
            loader = self.loader_train
            print('\nTraining: %d/%d epochs' % (self.current_epoch, self.max_epoch))
        else:
            net = self.net.eval()
            loader = self.loader_test
            print('\nTest:')

        # Stat.
        acc1 = 0
        acc5 = 0
        nb_instances = 0
        running_loss = 0
        delta = len(loader) // 3

        # Loop over mini-batches
        bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
        for it, batch in enumerate(tqdm(loader, bar_format=bar_format)):

            # Data
            sound = batch[0].to(self.dev, non_blocking=True)
            gt_class = batch[1].to(self.dev, non_blocking=True)

            # Get network outputs
            with torch.no_grad():
                sound = self.spectrogram(sound)
            if training:
                sound, gt_class = self.mixup(sound, gt_class)
                ######## Added for distillation
                if self.kd_loss_type == 'feat_reg' or self.kd_loss_type == 'at':
                    student_out = net(sound, out_feat=True)
                    pred_class = student_out[-1]
                #############
                else:
                    pred_class = net(sound)
            else:
                with torch.no_grad():
                    pred_class = net(sound)

            # Loss & backprop.
            if training:
                self.optim.zero_grad()
            loss_class = self.CEloss(pred_class, gt_class)

            if training:
                ######## Added for distillation
                if self.kd_loss_type == 'kl':
                    kd_loss = self.Distill([pred_class], sound)
                    loss = (1 - self.kd_alpha) * loss_class + self.kd_alpha * self.T ** 2 * kd_loss[0]
                elif self.kd_loss_type == 'feat_reg' or self.kd_loss_type == 'at':
                    kd_loss = self.Distill(student_out, sound)
                    loss = (1 - self.kd_alpha) * loss_class + self.kd_alpha * self.T ** 2 * kd_loss[0] + self.beta * \
                           kd_loss[1]
                else:
                    loss = loss_class
                ##############
                loss.backward()
                self.optim.step()

            # Log
            if training:
                temp = accuracy(pred_class, gt_class.max(1)[1])
            else:
                temp = accuracy(pred_class, gt_class)

            acc1 += temp[0]
            acc5 += temp[1]
            nb_instances += len(gt_class)
            running_loss += loss_class.item()
            if it % delta == delta - 1:
                self.print_log(running_loss, it, acc1, acc5, nb_instances)

        # Print log
        self.print_log(running_loss, it, acc1, acc5, nb_instances)
        header = 'Train' if training else 'Test'
        self.writer.add_scalar(header + '/loss', running_loss / (it + 1), self.current_epoch + 1)
        self.writer.add_scalar(header + '/acc1', 100 * acc1 / nb_instances, self.current_epoch + 1)
        self.writer.add_scalar(header + '/acc5', 100 * acc5 / nb_instances, self.current_epoch + 1)

    def load_state(self):
        ckpt = torch.load(
            self.path_to_ckpt,
            map_location=torch.device(self.dev)
        )
        self.net.load_state_dict(ckpt['net'])
        self.optim.load_state_dict(ckpt['optim'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.current_epoch = ckpt['epoch'] + 1
        # Check config is the same
        for key in ckpt['config'].keys():
            assert key in config.keys()
            assert config[key] == ckpt['config'][key]

    def save_state(self):
        dict_to_save = {
            'epoch': self.current_epoch,
            'net': self.net.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': config,
        }
        torch.save(dict_to_save, self.path_to_ckpt)

    def train(self):
        for _ in range(self.current_epoch, self.max_epoch):
            self.one_epoch(training=True)
            self.scheduler.step()
            self.one_epoch(training=False)
            self.save_state()
            self.current_epoch += 1
        print('Finished Training')

    def eval(self):
        self.one_epoch(training=False)


if __name__ == '__main__':

    # ---
    name_config = sys.argv[1].replace('.py', '')
    config = __import__(name_config, fromlist=['configs']).config
    # config = __import__('configs.'+name_config, fromlist=[None]).config
    print(config)

    config['mixup_alpha'] = config.get('mixup_alpha', 0.2)

    path2log = '/root/no_backup/dcase_challenge/mixup/' + name_config
    os.makedirs(path2log, exist_ok=True)
    make_tarfile(path2log + '/src.tgz', os.path.dirname(os.path.realpath(__file__)))

    # ---
    half_precision = False
    if config.get('half_precision', None):
        half_precision = True

    dcase_root = '/datasets_master/DCASE/'
    train_dataset = DCaseDataset(
        dcase_root + '2020/Task1/TAU-urban-acoustic-scenes-2020-mobile-development/',
        split='train',
    )
    test_dataset = DCaseDataset(
        dcase_root + '2020/Task1/TAU-urban-acoustic-scenes-2020-mobile-development/',
        split='val',
    )
    loader_train = DataLoader(
        train_dataset,
        batch_size=config['batchsize'],
        shuffle=True,
        pin_memory=True,
        num_workers=config['num_workers'],
        drop_last=True
    )
    loader_test = DataLoader(
        test_dataset,
        batch_size=config['batchsize'],
        shuffle=False,
        pin_memory=True,
        num_workers=config['num_workers'],
        drop_last=False
    )

    # ---
    if config.get('path_pretrain') is not None:
        print('Loading pretrained model in')
        path2load = config['path_pretrain']
        print(path2load)
        net = get_net[config['net']](config['dropout'], config['specAugment'], nb_classes=1000)
        ckpt = torch.load(path2load, map_location='cpu')
        net.load_state_dict(ckpt['net'])
        net.fc2 = nn.Linear(net.fc2.in_features, 10, bias=True)
    else:
        net = get_net[config['net']](config['dropout'], config['specAugment'])

    teacher_net = get_net[config['teacher_net']](0, None)
    teacher_net.load_state_dict(torch.load(config['teacher_path'])['net'])

    spectrogram = LogMelSpectrogram()
    if half_precision:
        print("Using FP16")
        net = net.half()
        # for layer in net.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.float()
        teacher_net = teacher_net.half()
        # for layer in teacher_net.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.float()

    print(net)
    print('Nb. of parameters: ', sum([p.numel() for p in net.parameters()]) / 1e3, 'k')

    # ---
    optim = torch.optim.AdamW(
        [
            {'params': net.parameters()},
        ],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        config['max_epoch'],
        eta_min=config['eta_min'],
    )

    # --- Training
    mng = TrainingManager(
        net,
        spectrogram,
        loader_train,
        loader_test,
        optim,
        scheduler,
        config['max_epoch'],
        path=path2log,
        teacher_net=teacher_net,
        kd_loss_type=config['kd_loss_type'],
        kd_alpha=config['kd_alpha'],
        temperature=config['temperature'],
        kd_beta=config['kd_beta'],
        mixup_alpha=config['mixup_alpha'],
        reload_ckpt=config['reload'],
    )
    mng.train()
