
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start = 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    max_iter = 20000,
    dataset='duckietown',
    im_root='./datasets/duckietown',
    train_im_anns='./datasets/duckietown/train.txt',
    val_im_anns='./datasets/duckietown/val.txt',
    n_classes=5,
    scales=[0.25, 2.],
    cropsize=[512, 1024],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
