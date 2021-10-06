import tensorflow as tf
import os
import matplotlib.pyplot as pl
import numpy as np

from generator import DataGenerator
from model import create_model, loss_asc, loss_dc
from configs import *
from params import *

""" 
Main script implements model training for all configurations discussed in the paper
"""


def scheduler(epoch: int, lr: float) -> float:
    """ Learning rate scheduler
    Args:
        epoch (int): Epoch number
        lr (float): Current learning rate
    Returns:
        lr (float): Next learning rate
    """
    if epoch % 50 == 0 and epoch > 0:
        return lr * 0.5
    else:
        return lr


def plot_log_loss_functions(fn_img: str, history, eps:float=0.0001):
    """ Plot log loss functions (L, L_A, L_D)
    Args:
        fn_img (str): Image filename
        history: Training history returned by the fit() function
        eps (float): very small value
    """
    pl.figure()
    pl.plot(np.log(np.array(history.history['loss'])+eps), label='$L$')
    pl.plot(np.log(np.array(history.history['asc_loss'])+eps), label='$L_A$')
    pl.plot(np.log(np.array(history.history['dc_loss'])+eps), label='$L_D$')
    pl.ylabel('Log loss')
    pl.xlabel('Epochs')
    pl.legend()
    pl.grid(True)
    pl.tight_layout()
    pl.savefig(fn_img, dpi=300)
    pl.close()


if __name__ == '__main__':

    # input shape for the network
    shape_in = (batch_size, 64, 431, 1)

    n_configs = len(configs)

    for c, curr_config in enumerate(configs):

        print('='*60)
        print(c+1, '/', n_configs, ':', curr_config['label'])
        print('='*60)

        if curr_config["do_mezza"]:
            print('Mezza normalization only applied during inference -> no separate model to be trained, we assume that',
                  'another configuration exists without Mezza normalization having the same settings otherwise')
            continue

        # create config-specific models
        model = create_model(shape_in=shape_in,
                             n_output_dim=10,
                             do_masking=curr_config["do_masking"],
                             emb_dim=emb_dim)

        # create config-specific data generator
        dg_train = DataGenerator(ds_type='training',
                                 dir_feat=dir_feat,
                                 batch_size=batch_size,
                                 do_masking=curr_config["do_masking"],
                                 use_td=curr_config["use_td"],
                                 asc_and_dc=curr_config["asc_and_dc"],
                                 n_batches_per_epoch=n_batches_per_epoch)

        # filenames of model and log file
        fn_model = os.path.join(dir_results, 'model_{}.h5'.format(curr_config["label"]))
        fn_log = fn_model.replace('.h5', '_training.txt')

        # training callbacks
        callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                     tf.keras.callbacks.ModelCheckpoint(filepath=fn_model,
                                                        monitor='loss',
                                                        save_best_only=True,
                                                        save_weights_only=True),
                     tf.keras.callbacks.CSVLogger(fn_log)]

        # compile model with two output branches and two designated loss functions (depending on config)
        model.compile(loss=[loss_asc(do_masking=curr_config["do_masking"],
                                     use_td_targets=curr_config["use_td_targets"],
                                     asc_and_dc=curr_config["asc_and_dc"],
                                     loss_weight_var=loss_weight_var),
                            loss_dc(do_masking=curr_config["do_masking"],
                                    use_td_targets=curr_config["use_td_targets"],
                                    asc_and_dc=curr_config["asc_and_dc"],
                                    loss_weight_var=loss_weight_var,
                                    loss_weight=loss_weight_dc)],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      metrics='acc')

        # train model
        history = model.fit(dg_train,
                            epochs=n_epochs,
                            callbacks=callbacks)

        # plot loss functions
        fn_img = fn_model.replace('.h5', '_loss_curve.png')
        plot_log_loss_functions(fn_img, history)

    # copy all results (stored on the GPU machine) to a designated folder on avdata for inspection
    cmd = 'cp /mnt/IDMT-WORKSPACE/DATA-STORE/abr/asc_2021_domain_adaptation/results_audasc/* /home/avdata/xtract/abr/2021_icassp_asc_da/'
    print(cmd)
    os.system(cmd)

    print('done with training :)')