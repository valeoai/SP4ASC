config = {
    "batchsize": 32,
    "num_workers": 4,
    "reload": False,
    "net": "Cnn6_60k",
    "dropout": 0.2,
    "specAugment": [128, 2, 16, 2],
    "lr": 1e-3,
    "eta_min": 1e-5,
    "max_epoch": 200,
    "weight_decay": 1e-5,
    "mixup_alpha": 0.2,
    "out_dir": "/root/no_backup/dcase_challenge/final_submission/",
}
