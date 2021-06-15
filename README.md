## DCASE21: Find a title

[**Find a title**]()  
[*Gilles Puy*](https://sites.google.com/site/puygilles/home),
[*Himalaya Jain*](https://himalayajain.github.io/),
[*Andrei Bursuc*](https://abursuc.github.io/)  
*valeo.ai, Paris, France*

If you find this code useful, please cite our [technical report]():
```
@article{vai21dcase,
  title={{DCASE}: Find a title},
  author={Puy, Gilles and Jain, Himalaya and Bursuc, Andrei},
  year={2021}
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
```/path/to/ADD_NAME_OF_REPO/Dockerfile```.
 
### Installation
1. Clone the repo:
```bash
$ git clone https://github.com/valeoai/ADD_NAME_OF_REPO
```

2. Install this repository:
```bash
$ pip install -e /path/to/ADD_NAME_OF_REPO
```
You can edit the code on the fly and import function and classes of dcase21 in other project as well.

3. If needed, you can uninstall this package by typing:
```bash
$ pip uninstall dcase21
```

### DCASE21 Datasets
If not already done, please download the development and evaluation datasets from
[here](http://dcase.community/challenge2021/task-acoustic-scene-classification#download). 

We suppose that these datasets are stored in ```/path/to/ADD_NAME_OF_REPO/data``` , which should thus 
contains the following sub-directories:
```bash
/path/to/ADD_NAME_OF_REPO/data/TAU-urban-acoustic-scenes-2020-mobile-development/ 
/path/to/ADD_NAME_OF_REPO/data/TAU-urban-acoustic-scenes-2021-mobile-evaluation/
```

If the dataset are stored elsewhere on your system, you can create soft links as follows:
```bash
$ ln -s /path/to/TAU-urban-acoustic-scenes-2020-mobile-development/ /path/to/ADD_NAME_OF_REPO/data/
$ ln -s /path/to/TAU-urban-acoustic-scenes-2021-mobile-evaluation/ /path/to/ADD_NAME_OF_REPO/data/
```


## Running the code

### Testing

A trained model is available in ```/path/to/ADD_NAME_OF_REPO/dcase21/trained_models/```.

To evaluate this model, type:
```bash
$ cd /path/to/ADD_NAME_OF_REPO/
$ python test.py
```

This model was trained and saved using 32-bit floats. 
This model is compressed in `test.py` by combining all convolutional and batchnorm layers to reach 62'474 parameters
and quantized using 16-bit floats. 

### Training

A script to train ... is available in . 
By default, the model and tensorboard logs are stored in `/path/to/ADD_NAME_OF_REPO/experiments`.
These ... epochs takes about ... hours to complete on a ...

### Using dcase21's models

Import dcase21 by typing
```python
from dcase21.models import cnn6_ours_60k
```

FLOT's constructor accepts one argument: `nb_iter`, which is the number of unrolled iterations of the Sinkhorn algorithm. In our experiments, we tested 1, 3, and 5 iterations. For example:
```python
flot = FLOT(nb_iter=3)
```

Input point clouds `pc1` and `pc2` can be passed to `flot` to estimate the flow from `pc1` to `pc2` as follows:
```python
scene_flow = flot([pc1, pc2])
```
The input point clouds `pc1` and `pc2` must be torch tensors of size `batch_size x nb_points x 3`.

## Acknowledgements
We are grateful to the authors of to have made their code publicly available.

## License
The repository is released under the [Apache 2.0 license]
