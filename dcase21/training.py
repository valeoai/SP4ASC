import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class CELoss:
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes

    def __call__(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        return -(pred * target).sum(1).mean()


class MixUp:
    def __init__(self, alpha, nb_classes):
        self.nb_classes = nb_classes
        if alpha is None:
            self.beta = None
        else:
            self.beta = torch.distributions.beta.Beta(alpha, alpha)
        self.training = None

    @staticmethod
    def mix(x, mix, ind):
        return x * mix + x[ind] * (1 - mix)

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, input, target):
        if self.training is None:
            raise ValueError("Choose training or testing mode")
        # Transform to one hot vector
        target = F.one_hot(target, num_classes=self.nb_classes)
        # Mix signals
        if self.beta is not None and self.training:
            ind = torch.randperm(input.shape[0])
            mix = self.beta.sample()
            input = MixUp.mix(input, mix, ind)
            target = MixUp.mix(target, mix, ind)
        else:
            pass
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
        config,
        path_to_ckpt,
        nb_classes=10,
    ):

        # Optim. methods
        self.optim = optim
        self.scheduler = scheduler

        # Dataloaders
        self.max_epoch = config["max_epoch"]
        self.loader_train = loader_train
        self.loader_test = loader_test

        # Networks
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net.to(self.dev)
        self.spectrogram = spectrogram.to(self.dev).eval()

        # Mixup and loss
        self.loss = CELoss(nb_classes=nb_classes)
        self.mixup = MixUp(alpha=config["mixup_alpha"], nb_classes=nb_classes)

        # Checkpoints
        self.config = config
        self.path_to_ckpt = path_to_ckpt + "/ckpt.pth"
        if config["reload"]:
            self.load_state()
        else:
            self.current_epoch = 0

        # Monitoring
        self.writer = SummaryWriter(
            path_to_ckpt + "/tensorboard/",
            purge_step=self.current_epoch + 1,
        )

    def print_log(self, running_loss, nb_it, acc, nb_instances):
        log = (
            "\nEpoch: {0:d} :".format(self.current_epoch)
            + " loss = {0:.3f}".format(running_loss / (nb_it + 1))
            + " - acc1 = {0:.3f}".format(100 * acc / nb_instances)
        )
        print(log)

    def one_epoch(self, training):

        # Train or eval mode
        if training:
            self.net.train()
            self.mixup.train()
            loader = self.loader_train
            print("\nTraining: %d/%d epochs" % (self.current_epoch, self.max_epoch))
        else:
            self.net.eval()
            self.mixup.eval()
            loader = self.loader_test
            print("\nTest:")

        # Stat.
        acc = 0
        nb_instances = 0
        running_loss = 0
        delta = len(loader) // 3

        # Loop over mini-batches
        bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
        for it, batch in enumerate(tqdm(loader, bar_format=bar_format)):

            # Data
            sound = batch[0].to(self.dev, non_blocking=True)
            gt_class = batch[1].to(self.dev, non_blocking=True)

            # Get network outputs with mixup during training
            with torch.no_grad():
                sound = self.spectrogram(sound)
                sound, gt_class = self.mixup(sound, gt_class)
                if not training:
                    pred_class = self.net(sound)
            if training:
                self.optim.zero_grad()
                pred_class = self.net(sound)

            # Loss & backprop
            loss_class = self.loss(pred_class, gt_class)
            if training:
                loss_class.backward()
                self.optim.step()

            # Log
            acc += (pred_class.max(1)[1] == gt_class.max(1)[1]).sum()
            nb_instances += gt_class.shape[0]
            running_loss += loss_class.item()
            if it % delta == delta - 1:
                self.print_log(running_loss, it, acc, nb_instances)

        # Print log
        self.print_log(running_loss, it, acc, nb_instances)
        header = "Train" if training else "Test"
        self.writer.add_scalar(
            header + "/loss", running_loss / (it + 1), self.current_epoch + 1
        )
        self.writer.add_scalar(
            header + "/acc", 100 * acc / nb_instances, self.current_epoch + 1
        )

    def load_state(self):
        ckpt = torch.load(self.path_to_ckpt, map_location=torch.device(self.dev))
        self.net.load_state_dict(ckpt["net"])
        self.optim.load_state_dict(ckpt["optim"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.current_epoch = ckpt["epoch"]
        # Check config is the same
        for key in ckpt["config"].keys():
            assert key in self.config.keys()
            if key == "reload":
                pass
            assert (
                self.config[key] == ckpt["config"][key]
            ), "Config file is not compatible with saved one."

    def save_state(self):
        dict_to_save = {
            "epoch": self.current_epoch,
            "net": self.net.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(dict_to_save, self.path_to_ckpt)

    def train(self):
        for _ in range(self.current_epoch, self.max_epoch):
            self.one_epoch(training=True)
            self.scheduler.step()
            self.one_epoch(training=False)
            self.current_epoch += 1
            self.save_state()
        print("Finished Training")

    def eval(self):
        self.one_epoch(training=False)
