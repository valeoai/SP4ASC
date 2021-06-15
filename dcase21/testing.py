import csv
import numpy as np
import os
import torch
import yaml
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm


class TestManager:
    def __init__(
        self,
        net,
        spectrogram,
        val_dataset,
        test_dataset,
        path2model,
    ):
        """

        :param net:
        :param spectrogram:
        :param val_dataset:
        :param test_dataset:
        :param path2model:
        """

        # Dataloaders
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
            num_workers=6,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
            num_workers=6,
            drop_last=False,
        )

        # Networks
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net.to(self.dev).eval()
        self.spectrogram = spectrogram.to(self.dev).eval()

        # Checkpoints
        self.path_to_ckpt = path2model + "/ckpt.pth"

    def save_predictions(self, basename_results, y_pred, filenames):
        """

        :param basename_results:
        :param y_pred:
        :param filenames:
        :return:
        """
        # Open
        csvfile = open(basename_results, "w", newline="")
        resultwriter = csv.writer(csvfile, delimiter="\t")

        # Header
        classes = ["filename", "scene_label"] + list(
            self.test_dataset.labelind2name.values()
        )
        resultwriter.writerow(classes)

        # Predicted class + Probabilities
        for i in range(y_pred.shape[0]):
            line = [filenames[i].replace("audio/", "")]
            line += [self.test_dataset.labelind2name[np.argmax(y_pred[i])]]
            line += [float(p) for p in y_pred[i]]
            resultwriter.writerow(line)

        # Close
        csvfile.close()

    def get_classwise_perf(self, y_pred, y_gt, dict_results):
        """

        :param y_pred:
        :param y_gt:
        :param dict_results:
        :return:
        """
        # Compute class-wise scores
        nb_class = y_pred.shape[1]
        class_acc = np.zeros(nb_class)
        class_loss = np.zeros(nb_class)
        for ind_class in range(nb_class):
            where = y_gt == ind_class
            class_loss[ind_class] = log_loss(
                y_true=y_gt[where], y_pred=y_pred[where], labels=list(range(nb_class))
            )
            class_acc[ind_class] = (
                np.argmax(y_pred[where], axis=1) == y_gt[where]
            ).mean() * 100
            #
            class_name = self.val_dataset.labelind2name[ind_class]
            dict_results["results"]["development_dataset"]["class_wise"][class_name][
                "logloss"
            ] = float(class_loss[ind_class])
            dict_results["results"]["development_dataset"]["class_wise"][class_name][
                "accuracy"
            ] = float(class_acc[ind_class])

        # Compute global scores
        global_acc = np.mean(class_acc)
        global_loss = np.mean(class_loss)

        # Store global scores
        print("\n\nGlobal performance:")
        dict_results["results"]["development_dataset"]["overall"]["logloss"] = float(
            global_loss
        )
        dict_results["results"]["development_dataset"]["overall"]["accuracy"] = float(
            global_acc
        )
        print(
            " " * 3,
            "log_loss: ",
            f"{global_loss:.3f}",
            "- Acc: ",
            f"{global_acc:.3f}",
        )
        print()

        # Print class-wise scores
        print("\nClass-wise performance:")
        max_shift = np.max(
            [len(lname) for lname in self.val_dataset.labelind2name.values()]
        )
        for ind_class in range(nb_class):
            class_name = self.val_dataset.labelind2name[ind_class]
            print(
                " " * 3,
                class_name,
                " " * (max_shift - len(class_name) + 1),
                "- log_loss: ",
                f"{class_loss[ind_class]:.3f}",
                "- Acc: ",
                f"{class_acc[ind_class]:.3f}",
            )
        print()

        return dict_results

    def get_devicewise_perf(self, y_pred, y_gt, devices, dict_results):
        """

        :param y_pred:
        :param y_gt:
        :param devices:
        :param dict_results:
        :return:
        """
        # List of devices
        list_devices = np.unique(devices)

        # Compute device-wise scores
        device_acc = np.zeros(len(list_devices))
        device_loss = np.zeros(len(list_devices))
        for ind_dev, dev_name in enumerate(list_devices):
            where = devices == dev_name
            device_loss[ind_dev] = log_loss(
                y_true=y_gt[where],
                y_pred=y_pred[where],
                labels=list(range(y_pred.shape[1])),
            )
            device_acc[ind_dev] = (
                np.argmax(y_pred[where], axis=1) == y_gt[where]
            ).mean() * 100
            #
            dict_results["results"]["development_dataset"]["device_wise"][dev_name][
                "logloss"
            ] = float(device_loss[ind_dev])
            dict_results["results"]["development_dataset"]["device_wise"][dev_name][
                "accuracy"
            ] = float(device_acc[ind_dev])

        # Write scores
        print("\nDevice-wise performance:")
        max_shift = np.max([len(dname) for dname in list_devices])
        for ind_dev, dev_name in enumerate(list_devices):
            print(
                " " * 3,
                dev_name,
                " " * (max_shift - len(dev_name) + 1),
                "- log_loss: ",
                f"{device_loss[ind_dev]:.3f}",
                "- Acc: ",
                f"{device_acc[ind_dev]:.3f}",
            )

        return dict_results

    @torch.no_grad()
    def one_epoch(self, split, basename_results, nb_augmentations):
        """

        :param split:
        :param basename_results:
        :param nb_augmentations:
        :return:
        """
        # Force augmentation at test time
        if nb_augmentations > 0:
            self.net.spec_augmenter.train()

        # List to store results
        y_pred, y_gt, filenames, devices = [], [], [], []

        # Loop over mini-batches
        loader = self.val_loader if split == "val" else None
        loader = self.test_loader if split == "test" else loader
        bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
        for it, batch in enumerate(tqdm(loader, bar_format=bar_format)):

            # Data
            sound = batch[0].to(self.dev, non_blocking=True)
            filenames.extend(batch[2])
            if split == "val":
                y_gt.append(batch[1])
                devices.append(batch[3])

            # Get network outputs
            sound = self.spectrogram(sound)
            if nb_augmentations > 0:
                pred_class = 0
                for _ in range(nb_augmentations):
                    pred_class += torch.softmax(self.net(sound), dim=1)
                pred_class = pred_class / nb_augmentations
            else:
                pred_class = torch.softmax(self.net(sound), dim=1)

            # Log
            y_pred.append(pred_class.cpu().numpy())

        # Combine results
        y_pred = np.concatenate(y_pred, 0)
        if split == "val":
            y_gt = np.concatenate(y_gt, 0)
            devices = np.concatenate(devices, 0)
            assert y_gt.shape[0] == y_pred.shape[0]
            assert y_gt.shape[0] == devices.shape[0]
            assert y_pred.shape[0] == len(self.val_dataset)
        else:
            assert y_pred.shape[0] == len(self.test_dataset)

        # Compute performance metrics
        if split == "val":
            # Open default yaml file
            with open(
                os.path.dirname(os.path.abspath(__file__)) + "/default.yaml"
            ) as file:
                dict_results = yaml.load(file, Loader=yaml.FullLoader)
            # Results on validation set
            dict_results = self.get_classwise_perf(y_pred, y_gt, dict_results)
            dict_results = self.get_devicewise_perf(y_pred, y_gt, devices, dict_results)
            # Number of parameters
            dict_results["system"]["complexity"]["total_parameters"] = int(
                self.net.get_nb_parameters()
            )
            dict_results["system"]["complexity"]["total_parameters_non_zero"] = int(
                self.net.get_nb_parameters()
            )
            dict_results["system"]["complexity"]["model_size"] = float(
                self.net.get_nb_parameters() * 16 / 8 / 1024
            )
            # Label of submission
            dict_results["submission"]["label"] = basename_results
            # Number of TTA
            dict_results["system"]["description"]["ensemble_method_subsystem_count"] = \
                int(nb_augmentations) if nb_augmentations > 0 else 1
            # Save results
            with open(basename_results + ".meta.yaml", "w") as file:
                yaml.dump(dict_results, file)
                file.close()

        # Save predictions
        if split == "test":
            self.save_predictions(basename_results + ".output.csv", y_pred, filenames)

    def load_state(self):
        """

        :return:
        """
        # Load checkpoint
        param = torch.load(
            self.path_to_ckpt,
            map_location=torch.device(self.dev)
        )["net"]
        self.net.load_state_dict(param)
        print("Model in " + self.path_to_ckpt + " loaded.")

        # Merge batch norms and convolution layers
        self.net.merge_conv_bn()
        print("\n\nCompressed network")
        print(self.net)
        print(
            "Nb. of parameters of compressed network: ",
            self.net.get_nb_parameters() / 1e3,
            "k",
        )

        # Simulate quantization on disk in float16 and reloading in float32
        self.net = self.net.half()
        self.net = self.net.float()
        print("Model quantized")

    def test(self, basename_results=None, nb_augmentations=0):
        """

        :param basename_results:
        :param nb_augmentations:
        :return:
        """
        self.load_state()
        print("\n\nEvaluate on validation set")
        self.one_epoch(
            "val",
            basename_results=basename_results,
            nb_augmentations=nb_augmentations,
        )
        print("\n\nRun on test set")
        self.one_epoch(
            "test",
            basename_results=basename_results,
            nb_augmentations=nb_augmentations,
        )
