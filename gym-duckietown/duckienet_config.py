inet_cfg = dict(
    model_dir="/path_to_repo/intention_net/new_model",
    data_dir="/path_to_repo/intention_net/data",
    intention_mode="LPE_NO_SIAMESE",
    input_frame="NORMAL",
    directory="/path_to_repo/intention_net/intention_net"
)

bisenet_cfg = dict(
    n_classes=5,
    weights_path="/path_to_repo/bisenet/model_final.pth",
    directory="/path_to_repo/bisenet"
)
