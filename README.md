## Separable convolutions and test-time augmentations for low-complexity and calibrated acoustic scene classification (DCASE21 Challenge)

[**SEPARABLE CONVOLUTIONS AND TEST-TIME AUGMENTATIONS FOR LOW-COMPLEXITY AND CALIBRATED ACOUSTIC SCENE CLASSIFICATION**]() 

[*Gilles Puy*](https://sites.google.com/site/puygilles/home),
[*Himalaya Jain*](https://himalayajain.github.io/),
[*Andrei Bursuc*](https://abursuc.github.io/)  
*valeo.ai, Paris, France*

This repo contains the code to reproduce the results of the systems we submitted to the Task1a of the DCASE21 challenge. 
Please refer to [link1](http://dcase.community/challenge2021/task-acoustic-scene-classification#subtask-a) and 
[link2](https://arxiv.org/abs/2105.13734) for more information about the challenge.


If you find this code useful, please cite our [technical report]():
```
@techreport{vai21dcase,
  title={Separable convolutions and test-time augmentations for low-complexity and calibrated acoustic scene classification},
  author={Puy, Gilles and Jain, Himalaya and Bursuc, Andrei},
  institution={{DCASE2021 Challenge}},
  year={2021},
}
```


## Preparation

### Environment
* Python >= 3.7
* CUDA >= 10.2
```bash
$ pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install tqdm scikit-learn tensorboard pandas pyaml torchlibrosa
$ apt install -y libsndfile1
```

To help you re-create this environment, we also provide the dockerfile used to run the experiments in 
```/path/to/SP4ASC/Dockerfile```.
 
### Installation
1. Clone the repo:
```bash
$ git clone https://github.com/valeoai/SP4ASC
```

2. Optional. Install this repository:
```bash
$ pip install -e /path/to/SP4ASC
```
You can edit the code on the fly and import function and classes of sp4asc in other project as well.

3. If needed, you can uninstall this package by typing:
```bash
$ pip uninstall sp4asc
```

### DCASE21 Datasets
If not already done, please download the development and evaluation datasets from
[here](http://dcase.community/challenge2021/task-acoustic-scene-classification#download). 

We suppose that these datasets are stored in ```/path/to/SP4ASC/data``` , which should thus 
contains the following sub-directories:
```bash
/path/to/SP4ASC/data/TAU-urban-acoustic-scenes-2020-mobile-development/ 
/path/to/SP4ASC/data/TAU-urban-acoustic-scenes-2021-mobile-evaluation/
```

If the dataset are stored elsewhere on your system, you can create soft links as follows:
```bash
$ mkdir /path/to/SP4ASC/data/
$ ln -s /path/to/TAU-urban-acoustic-scenes-2020-mobile-development/ /path/to/SP4ASC/data/
$ ln -s /path/to/TAU-urban-acoustic-scenes-2021-mobile-evaluation/ /path/to/SP4ASC/data/
```


## Running the code

### Testing

Our trained models are available in ```/path/to/SP4ASC/trained_models/```. These models were trained and saved using 32-bit floats. Each model is compressed in `test.py` by combining all convolutional and batchnorm layers to reach 62'474 parameters
and quantized using 16-bit floats. 

1. The results of the model trained with cross entropy and without mixup can be reproduced by typing:
```bash
$ cd /path/to/SP4ASC/
$ python test.py --config configs/cnn6_small_dropout_2_specAugment_128_2_32_2.py --nb_aug 30
```

2. The results of the model trained with cross entropy and mixup can be reproduced by typing:
```bash
$ cd /path/to/SP4ASC/
$ python test.py --config configs/cnn6_small_dropout_2_specAugment_128_2_16_2_mixup_2.py --nb_aug 30
```

3. The results of the model trained with focal loss can be obtained by typing:
```
$ cd /path/to/SP4ASC/
$ python test.py --config configs/cnn6_small_dropout_2_specAugment_128_2_32_2_focal_loss.py --nb_aug 30
```
Note that we have retrained this model since the submission to the challenge. The log loss (metric used to 
rank the systems in the challenge) is unchanged compared to the submitted model. We observe a slight variation 
for the top-1 accuracy.

The performance without test-time augmentations (`--nb_aug 0`) are:

|                  |  Log loss  |  Accuracy   |
|---               |---         |---          |
|Model submitted   |    0.95    |  66.7       |               
|This model        |    0.94    |  67.1       |

The performance without test-time augmentations (`--nb_aug 30`) are:

|                  |  Log loss  |  Accuracy   |
|---               |---         |---          |
|Model submitted   |    0.88    |  68.3       |               
|This model        |    0.89    |  67.0       |



### Training

A script to train a model with cross-entropy and mixup is available at ```/path/to/SP4ASC/train.py```.
This script should be called with a config file that sets the training parameters. An example of such a file is given ```configs/example.py```.

One can train a model by typing:
```bash
$ cd /path/to/SP4ASC/
$ python train.py --config configs/example.py
```

Once trained, this model can be evaluated by typing:
```bash
$ cd /path/to/SP4ASC/
$ python test.py --config configs/example.py --nb_aug 10
```

The argument `XX` after ```--nb_aug``` defines the number of augmentations done at test time. Set `XX` to `0` to remove all augmentations.


### Retraining the provided models

The training parameters used for each of the provided models can be found in ```/path/to/SP4ASC/configs```.

**Note:** *Retraining the model will erase the provided checkpoint in the directory 
```/path/to/SP4ASC/trained_model/.``` You can avoid this behaviour by copying the configs files in 
`configs/,` editing the field `-out_dir` in the copied file, and using this new file as an argument of 
`--config` below.*

1. The model trained with cross entropy and without mixup can be retrained by typing
```bash
$ cd /path/to/SP4ASC/
$ python train.py --config configs/cnn6_small_dropout_2_specAugment_128_2_32_2.py
```

2. The model trained with cross entropy and with mixup can be retrained by typing
```bash
$ cd /path/to/SP4ASC/
$ python train.py --config configs/cnn6_small_dropout_2_specAugment_128_2_16_2_mixup_2.py
```

3. The model trained with the focal loss can be retrained by typing
```bash
$ cd /path/to/SP4ASC/
$ python train.py --configs/cnn6_small_dropout_2_specAugment_128_2_32_2_focal_loss.py
```


### Using sp4asc model

You can reuse sp4asc in your own project by first installing this package (see above).
Then, import the model by typing
```python
from sp4asc.models import Cnn6_60k
```

The constructor takes two arguments `dropout`, a scalar indicating the dropout rate, and 
`spec_aug`, a list of the form `[T, n, F, m]`, where `T` is the size of the mask on the time axis,
`n` the number of mask of the time axis, `F` the size of the mask on the frequency axis, and `m` 
the number of masks on the frequency axis. 
```python
net = Cnn6_60k(dropout=0.2, spec_aug=[128, 2, 16, 2])
```


## Acknowledgements
Our architecture is based on CNN6 which is described in [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211).
We modified the original CNN6 architecture by using separable convolutions and changing the number channels per layer to meet the complexity constraints of the DCASE21 Task1a challenge.

The original implementation of CNN6 without separable convolutions is available [here](https://github.com/qiuqiangkong/audioset_tagging_cnn). 
We are grateful to the authors for providing this implementation.

We are also grateful to the authors of [torchlibrosa](https://github.com/qiuqiangkong/torchlibrosa) which we use to compute the log-mel spectrograms.

## License
The repository is released under the [Apache 2.0 license](./LICENSE)
