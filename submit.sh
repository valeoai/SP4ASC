#python test.py \
#	--config configs/cnn6_small_dropout_2_specAugment_128_2_32_2.py \
#	--nb_aug 30 \
#	--out_name puy_vai_task1a_1 \
#	--sys_name ce_tta

python test.py \
        --config configs/cnn6_small_dropout_2_specAugment_128_2_16_2_mixup_2.py \
        --nb_aug 30 \
        --out_name puy_vai_task1a_2 \
        --sys_name ce_mu_tta
