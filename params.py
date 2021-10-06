import platform

# set paths to dataset & feature storage folder
if "Windows" in platform.platform():
    raise NotImplementedError
else:
    # feature set location
    dir_feat = '/mnt/IDMT-WORKSPACE/DATA-STORE/abr/asc_2021_audasc_features'
    # result directory to save models / results
    dir_results = '/mnt/IDMT-WORKSPACE/DATA-STORE/abr/asc_2021_domain_adaptation/results_audasc'

# parameters
emb_dim = 256  # same size as final two dense layer in the "Kaggle" model
batch_size = 512
n_batches_per_epoch = 100
n_epochs = 300
emb_norm_type = "layer_norm"

# loss weights
loss_weight_asc = 1
loss_weight_dc = 10
loss_weight_var = 500
