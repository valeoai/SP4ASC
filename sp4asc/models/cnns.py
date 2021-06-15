"""
This original CNN6 model without separable convolution can be found in
https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py.
"""
import torch
import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        window="hann",
        center=True,
        pad_mode="reflect",
        ref=1.0,
        amin=1e-10,
        top_db=None,
        mel_bins=256,
        sample_rate=44100,
        fmin=0,
        fmax=22050,
        window_size=2048,
        hop_size=1024,
    ):
        """

        :param window:
        :param center:
        :param pad_mode:
        :param ref:
        :param amin:
        :param top_db:
        :param mel_bins:
        :param sample_rate:
        :param fmin:
        :param fmax:
        :param window_size:
        :param hop_size:
        """
        super(LogMelSpectrogram, self).__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

    def forward(self, x):
        """
        Return logmel spectrogram of x
        :param x: size (batch_size, length)
        :return: size (batch_size, 1, time_steps, nb_mel_bins)
        """
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        return x


class Conv(nn.Module):
    def __init__(self, cout):
        """

        :param cout:
        """
        super(Conv, self).__init__()
        self.bn = nn.BatchNorm2d(cout, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2)

    def conv(self, x):
        """

        :param x:
        :return:
        """
        raise NotImplementedError()

    def merge_conv_bn(self):
        """

        :return:
        """
        raise NotImplementedError()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SepConv(Conv):
    def __init__(self, cin, cout):
        """

        :param cin:
        :param cout:
        """
        super(SepConv, self).__init__(cout)

        # --- 1 x 1 conv
        self.ch_conv1 = nn.Conv2d(
            cin,
            cout,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )

        # --- Spatial convolution channel-wise convolution
        # 1 x 3 with dilation 2
        self.sp_conv1 = nn.Conv2d(
            cout,
            cout,
            kernel_size=(3, 1),
            stride=1,
            padding=(2, 0),
            dilation=(2, 1),
            groups=cout,
            bias=False,
            padding_mode="zeros",
        )
        # 3 x 1 with dilation 2
        self.sp_conv2 = nn.Conv2d(
            cout,
            cout,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 2),
            dilation=(1, 2),
            groups=cout,
            bias=False,
            padding_mode="zeros",
        )

    def merge_conv_bn(self):
        """

        :return:
        """
        # Compute scale and shift from batch norm parameters
        scale = self.bn.weight / torch.sqrt(self.bn.running_var)
        bias = self.bn.bias - scale * self.bn.running_mean
        # Update convolution layers
        self.sp_conv1.weight.data = (
            self.sp_conv1.weight.data * scale[:, None, None, None].data
        )
        self.sp_conv2.weight.data = (
            self.sp_conv2.weight.data * scale[:, None, None, None].data
        )
        self.sp_conv2.bias = nn.Parameter(bias.detach())
        # Remove batch norm
        self.bn = nn.Identity()

    def conv(self, x):
        """

        :param x:
        :return:
        """
        x = self.ch_conv1(x)
        x = self.sp_conv1(x) + self.sp_conv2(x)
        return x


class Affine2D(nn.Module):
    def __init__(self, cin):
        """

        :param cin:
        """
        super(Affine2D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, cin, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, cin, 1, 1))

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.weight * x + self.bias


class Cnn(nn.Module):
    def __init__(
        self,
        mel_bins=256,
        channels=[64, 128, 128, 128],
        nb_classes=10,
        dropout=0,
        spec_augment=None,
    ):
        """

        :param mel_bins:
        :param channels:
        :param nb_classes:
        :param dropout:
        :param spec_augment:
        """
        super(Cnn, self).__init__()

        # Normalise mel spectrogram
        self.bn0 = nn.BatchNorm2d(mel_bins)

        # Spec augmenter
        if spec_augment is None:
            self.spec_augmenter = nn.Identity()
        else:
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=spec_augment[0],
                time_stripes_num=spec_augment[1],
                freq_drop_width=spec_augment[2],
                freq_stripes_num=spec_augment[3],
            )

        # Conv. net
        self.conv1 = SepConv(1, channels[0])
        self.conv2 = SepConv(channels[0], channels[1])
        self.conv3 = SepConv(channels[1], channels[2])
        self.conv4 = SepConv(channels[2], channels[3])

        # Classifier
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels[3], channels[3], bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(channels[3], nb_classes, bias=True)

    def merge_conv_bn(self):
        """

        :return:
        """

        # --- Replace first batch norm with affine layer
        # Compute scale and shift from batch norm parameters
        scale = self.bn0.weight / torch.sqrt(self.bn0.running_var)
        bias = self.bn0.bias - scale * self.bn0.running_mean
        # Define affine layer
        self.bn0 = Affine2D(scale.shape[0]).to(scale.device)
        self.bn0.weight.data[0, :, 0, 0] = scale.data
        self.bn0.bias.data[0, :, 0, 0] = bias.data

        # --- Process conv layers
        self.conv1.merge_conv_bn()
        self.conv2.merge_conv_bn()
        self.conv3.merge_conv_bn()
        self.conv4.merge_conv_bn()

    def get_nb_parameters(self):
        """

        :return:
        """
        p = [p.numel() for p in self.state_dict().values()]
        return sum(p)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        """Input size - (batch_size, 1, time_steps, mel_bins)  """

        # Normalise mel spectrogram
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # Masking
        x = self.spec_augmenter(x)

        # Convolutions
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)

        # Classifier
        x = torch.mean(feat4, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class Cnn6_60k(Cnn):
    def __init__(self, dropout, spec_aug):
        """

        :param dropout:
        :param spec_aug:
        """
        super(Cnn6_60k, self).__init__(
            dropout=dropout,
            nb_classes=10,
            channels=[64, 128, 128, 128],
            spec_augment=spec_aug,
        )
