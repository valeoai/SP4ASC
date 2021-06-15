import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class DCaseDataset(Dataset):
    """
    Dataloader for DCase dataset
    Structure of the class is taken from:
    https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/audio_classifier_tutorial.ipynb
    """

    labelind2name = {
        0: "airport",
        1: "bus",
        2: "metro",
        3: "metro_station",
        4: "park",
        5: "public_square",
        6: "shopping_mall",
        7: "street_pedestrian",
        8: "street_traffic",
        9: "tram",
    }
    name2labelind = {
        "airport": 0,
        "bus": 1,
        "metro": 2,
        "metro_station": 3,
        "park": 4,
        "public_square": 5,
        "shopping_mall": 6,
        "street_pedestrian": 7,
        "street_traffic": 8,
        "tram": 9,
    }

    def __init__(self, root_dir, split):
        """

        :param root_dir:
        :param split:
        """

        # Open csv files
        self.split = split
        self.root_dir = root_dir
        if split == "train":
            csv_path = root_dir + "/evaluation_setup/fold1_train.csv"
            meta_path = root_dir + "/meta.csv"
        elif split == "val":
            csv_path = root_dir + "/evaluation_setup/fold1_evaluate.csv"
            meta_path = root_dir + "/meta.csv"
        elif split == "test":
            csv_path = root_dir + "/evaluation_setup/fold1_test.csv"
            meta_path = None
        else:
            raise ValueError("Split not implemented")
        csvData = pd.read_csv(csv_path, sep="\t")
        metaData = pd.read_csv(meta_path, sep="\t") if meta_path is not None else None

        # In test mode, just get file list
        if split == "test":
            self.file_names = []
            for i in range(0, len(csvData)):
                self.file_names.append(csvData.iloc[i, 0])
            return

        # Lists of file names and labels
        self.file_names, self.labels = [], []
        for i in range(0, len(csvData)):
            self.file_names.append(csvData.iloc[i, 0])
            self.labels.append(csvData.iloc[i, 1])

        # Device for each audio file
        self.devices = {}
        for i in range(0, len(metaData)):
            self.devices[metaData.iloc[i, 0]] = metaData.iloc[i, 3]

        # Transform class name to index
        self.labels = [self.name2labelind[name] for name in self.labels]

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """

        # Load data
        filepath = self.root_dir + self.file_names[index]
        sound, sfreq = torchaudio.load(filepath, normalization=True)
        assert sound.shape[0] == 1, "Expected mono channel"
        sound = torch.mean(sound, dim=0)
        assert sfreq == 44100, "Expected sampling rate of 44.1 kHz"

        # Remove last samples if longer than expected
        if sound.shape[-1] >= 441000:
            sound = sound[:441000]

        if self.split == "test":
            return sound, 255, self.file_names[index], "unknown"
        else:
            return (
                sound,
                self.labels[index],
                self.file_names[index],
                self.devices[self.file_names[index]],
            )

    def __len__(self):
        return len(self.file_names)
