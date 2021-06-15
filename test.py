import argparse
import os
import yaml
from sp4asc.datasets.dcase import DCaseDataset
from sp4asc.models import get_net
from sp4asc.models.cnns import LogMelSpectrogram
from sp4asc.testing import TestManager


if __name__ == "__main__":

    # --- Args
    parser = argparse.ArgumentParser(description="Test on validation and test sets.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.py",
        help="Path to config file describing training parameters",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="result_task1a_1",
        help="Basename for file with results to submit",
    )
    parser.add_argument(
        "--sys_name",
        type=str,
        default="example",
        help="System_name",
    )
    parser.add_argument(
        "--nb_aug",
        type=int,
        default=0,
        help="Number of augmentation at test time",
    )
    args = parser.parse_args()

    # --- Load config file
    name_config = args.config.replace(".py", "").replace(os.path.sep, ".")
    config = __import__(name_config, fromlist=["config"]).config
    path2model = os.path.join(config["out_dir"], name_config)

    # --- Datasets
    current_dir = os.path.dirname(os.path.abspath(__file__))
    val_dataset = DCaseDataset(
        current_dir + "/data/TAU-urban-acoustic-scenes-2020-mobile-development/",
        split="val",
    )
    test_dataset = DCaseDataset(
        current_dir + "/data/TAU-urban-acoustic-scenes-2021-mobile-evaluation/",
        split="test",
    )

    # --- Load network
    spectrogram = LogMelSpectrogram()
    net = get_net[config["net"]](config["dropout"], config["specAugment"])
    print("\n\nNet at training time")
    print(net)
    print("Nb. of parameters at training time: ", net.get_nb_parameters() / 1e3, "k")

    # --- Test
    mng = TestManager(
        net,
        spectrogram,
        val_dataset,
        test_dataset,
        path2model=path2model,
    )
    mng.test(
        basename_results=args.out_name,
        nb_augmentations=args.nb_aug,
    )

    # --- Complete submission information
    # Augmentations used
    list_of_augmentations = 'SpecAugment'
    if config["mixup_alpha"] is not None:
        list_of_augmentations += ', mixup'
    #
    with open(args.out_name + ".meta.yaml", "r") as file:
        dict_results = yaml.load(file, Loader=yaml.FullLoader)
    #
    with open(args.out_name + ".meta.yaml", "w") as file:
        dict_results["system"]["description"]["data_augmentation"] = \
            list_of_augmentations
        dict_results["submission"]["name"] = args.sys_name
        dict_results["submission"]["abbreviation"] = args.sys_name
        yaml.dump(dict_results, file)
