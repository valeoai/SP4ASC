## DCASE21: Find a title

[**Find a title**]()  
[*Gilles Puy*](https://sites.google.com/site/puygilles/home),
[*Himalaya Jain*](https://himalayajain.github.io/),
[*Andrei Bursuc*](https://abursuc.github.io/)  
*valeo.ai, Paris, France*

This repo contains the code to reproduce the results of the systems we submitted to the Task1a of the DCASE21 challenge 
[link1](http://dcase.community/challenge2021/task-acoustic-scene-classification#subtask-a)[link2](https://arxiv.org/abs/2105.13734).


If you find this code useful, please cite our [technical report]():
```
@article{vai21dcase,
  title={{DCASE}: Find a title},
  author={Puy, Gilles and Jain, Himalaya and Bursuc, Andrei},
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
$ ln -s /path/to/TAU-urban-acoustic-scenes-2020-mobile-development/ /path/to/SP4ASC/data/
$ ln -s /path/to/TAU-urban-acoustic-scenes-2021-mobile-evaluation/ /path/to/SP4ASC/data/
```


## Running the code

### Testing

A trained model is available in ```/path/to/SP4ASC/trained_models/```.

To evaluate this model, type:
```bash
$ cd /path/to/SP4ASC/
$ python test.py
```

This model was trained and saved using 32-bit floats. 
This model is compressed in `test.py` by combining all convolutional and batchnorm layers to reach 62'474 parameters
and quantized using 16-bit floats. 

### Training

A script to train with cross-entropy and mixup (or not) is available at ```/path/to/SP4ASC/train.py```.
This script should be called with an associates config file. We provide in ```/path/to/SP4ASC/configs``` the config files used to obtained the models in ```/path/to/SP4ASC/trained_models/```.

For example, one can train a model by typing
```bash
$ cd /path/to/SP4ASC/
$ python train.py --config configs/cnn6...
```

Once trained, this model can be evaluated by typing
```bash
$ cd /path/to/SP4ASC/
$ python test.py --config configs/cnn6...
```

### Using sp4asc model

You can reuse sp4asc in your own project by first installing this package (see above).

Then import the model by typing
```python
from sp4asc.models import cnn6_ours_60k
```

```python
flot = FLOT(nb_iter=3)
```


## Acknowledgements
Our architecture is based on CNN6 which is described in [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211).
We modified the original CNN6 architecture by using separable convolutions and changing the number channels per layer to meet the complexity constraints of the DCASE21 Task1a challenge.

The original implementation of CNN6 without separable convolutions is available [here](https://github.com/qiuqiangkong/audioset_tagging_cnn). We are grateful to the authors of this work.

We are also grateful to the authors of [torchlibrosa](https://github.com/qiuqiangkong/torchlibrosa) which we use to compute the log-mel spectrograms.

## License
The repository is released under the [Apache 2.0 license]
