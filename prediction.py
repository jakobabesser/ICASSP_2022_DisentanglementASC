import numpy as np
import os
import platform
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as pl
import pandas as pd

from configs import *
from params import *
from generator import DataGenerator
from model import loss_asc, loss_dc, create_model
from normalization import mezza_normalization

""" Script to run inference of all trained models on the test data """

class_labels_asc = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian',
                    'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']

class_labels_dc = ['A', 'B', 'C']


class ASCPredictor:
    """ Prediction class """
    def __init__(self,
                 fn_model,
                 do_masking=False,
                 batch_size=512,
                 emb_dim=256):
        """ Initialize class
        Args:
            fn_model (str): Model filename (note that we only store the weights!, saving the full model
                            instead caused some errors with the custom loss functions)
            do_masking (bool): Switch to do embedding masking
            batch_size (int): Batch size
            emb_dim (int): Embedding size
        """
        self.emb_dim = emb_dim
        self.do_masking = do_masking

        shape_in = (batch_size, 64, 431, 1)

        # create model from scratch and load weights
        self.multi_branch_model = create_model(shape_in=shape_in,
                                               n_output_dim=10,
                                               do_masking=do_masking,
                                               emb_dim=self.emb_dim,
                                               inference=True)
        self.multi_branch_model.load_weights(fn_model)

        # double embedding size if embedding masking is used
        if do_masking:
            self.emb_dim *= 2

    def predict_feat(self, feat, do_activate_upper_embedding_half=True):
        """ Compute model prediction for given feature tensor
        Args:
            feat (4d np.ndarray): Feature tensor (batch x freq x time x channel)
            do_activate_upper_embedding_half (bool): If True and if masking is activated, binary mask is set such that
                                    embedding is optimized for ASC task
        Returns:
            prediction (2d np.ndarray): Prediction (batch x n_classes)
        """
        n_patches = feat.shape[0]
        mask = np.ones((n_patches, self.emb_dim))
        if self.do_masking:
            # if masking is selected, only consider first half of embedding (trained for ASC task)
            if do_activate_upper_embedding_half:
                mask[:, self.emb_dim//2:] = 0
            else:
                mask[:, :self.emb_dim//2] = 0

        # get output of dual-output model
        dual_predictions = self.multi_branch_model.predict((feat, mask))

        return dual_predictions


def create_cm_plot(fn_img, true, pred, labels):
    """ Create confusion matrix plot
    Args:
        fn_img (str): Image filename
        true (np.ndarray): True class indices
        pred (np.ndarray): Predicted class indices
    """
    cm = confusion_matrix(true, pred, normalize="true")
    tick_vals = np.arange(len(labels))
    pl.figure()
    pl.imshow(cm, aspect="equal")
    pl.xticks(tick_vals, labels, rotation=60)
    pl.yticks(tick_vals, labels)
    pl.tight_layout()
    pl.savefig(fn_img)
    pl.close()
    print(fn_img, ' saved')


if __name__ == '__main__':

    n_configs = len(configs)

    print("Import data")
    feat = {}
    asc_target = {}
    dc_target = {}
    is_source_domain = {}
    is_target_domain = {}

    # iterate over training and test set and import both
    for ds_type in ('training', 'test'):
        # let's use the data generator to import the data (but for nothing else)
        dg = DataGenerator(ds_type=ds_type,
                           dir_feat=dir_feat,
                           batch_size=batch_size)  # other parameters don't matter here

        # load data from generator
        feat[ds_type] = dg.feat
        asc_target[ds_type] = dg.asc_target
        dc_target[ds_type] = dg.dc_target
        is_source_domain[ds_type] = dc_target[ds_type][:, 0].astype(bool)
        is_target_domain[ds_type] = np.logical_not(is_source_domain[ds_type])

    print("Run predictions")
    for ds_type in ('training', 'test'):

        print('=' * 60)
        print(ds_type)
        print('=' * 60)

        # split into source (device A) and target (devices B & C)
        feat_source = feat[ds_type][is_source_domain[ds_type], :, :, :]
        feat_target = feat[ds_type][is_target_domain[ds_type], :, :, :]
        asc_true_s = asc_target[ds_type][is_source_domain[ds_type], :]
        asc_true_t = asc_target[ds_type][is_target_domain[ds_type], :]
        dc_true_s = dc_target[ds_type][is_source_domain[ds_type], :]
        dc_true_t = dc_target[ds_type][is_target_domain[ds_type], :]

        acc = np.zeros((n_configs, 8))

        # iterate over models and compute accuracy for source and target domain data in testset
        for c in range(n_configs):

            config_label = configs[c]["label"]

            print('-' * 60)
            print(c + 1, '/', n_configs, ':', config_label)
            print('-' * 60)

            fn_model = os.path.join(dir_results, 'model_{}.h5'.format(config_label))

            # if Mezza normalization is active, let's use the model trained with the same parameters
            fn_model = fn_model.replace('-mezza', '')

            # get prediction
            predictor = ASCPredictor(fn_model,
                                     do_masking=configs[c]["do_masking"],
                                     batch_size=batch_size)

            # Apply band-wise statistics matching procedure proposed by Mezza et al. (EUSIPCO 2021)
            if configs[c]["do_mezza"] and ds_type == "test":
                # get source domain data from training set
                feat_source_training = feat["training"][is_source_domain["training"], :, :, :]

                # adapt test data (target domain) to training data (source domain)
                feat_target[:, :, :, 0] = mezza_normalization(feat_target[:, :, :, 0],
                                                              feat_source_training[:, :, :, 0],
                                                              switch_time_freq=True)

            # get ASC predictions on source domain data
            asc_pred_s_u, dc_pred_s_u = predictor.predict_feat(feat_source, True)
            # get DC predictions on source domain data
            asc_pred_s_d, dc_pred_s_d = predictor.predict_feat(feat_source, False)
            # get ASC predictions on target domain data
            asc_pred_t_u, dc_pred_t_u = predictor.predict_feat(feat_target, True)
            # get DC predictions on target domain data
            asc_pred_t_d, dc_pred_t_d = predictor.predict_feat(feat_target, False)

            # accuracies
            acc[c, 0] = accuracy_score(np.argmax(asc_true_s, axis=1), np.argmax(asc_pred_s_u, axis=1))
            acc[c, 1] = accuracy_score(np.argmax(asc_true_s, axis=1), np.argmax(asc_pred_s_d, axis=1))
            acc[c, 2] = accuracy_score(np.argmax(asc_true_t, axis=1), np.argmax(asc_pred_t_u, axis=1))
            acc[c, 3] = accuracy_score(np.argmax(asc_true_t, axis=1), np.argmax(asc_pred_t_d, axis=1))

            acc[c, 4] = accuracy_score(np.argmax(dc_true_s, axis=1), np.argmax(dc_pred_s_u, axis=1))
            acc[c, 5] = accuracy_score(np.argmax(dc_true_s, axis=1), np.argmax(dc_pred_s_d, axis=1))
            acc[c, 6] = accuracy_score(np.argmax(dc_true_t, axis=1), np.argmax(dc_pred_t_u, axis=1))
            acc[c, 7] = accuracy_score(np.argmax(dc_true_t, axis=1), np.argmax(dc_pred_t_d, axis=1))

        # finally export CSV file with all metrics for all configs
        df_res = pd.DataFrame(data=acc, index=[_['label'] for _ in configs],
                              columns=('ASC-S-hi', 'ASC-S-lo', 'ASC-T-hi', 'ASC-T-lo', 'DC-S-hi',
                                       'DC-S-lo', 'DC-T-hi', 'DC-T-lo'))
        fn_csv = os.path.join(dir_results, 'acc_8col_{}.csv'.format(ds_type))
        df_res.to_csv(fn_csv, sep=';', float_format='%.2f')

    cmd = 'cp /mnt/IDMT-WORKSPACE/DATA-STORE/abr/asc_2021_domain_adaptation/results_audasc/* /home/avdata/xtract/abr/2021_icassp_asc_da/'
    print(cmd)
    os.system(cmd)

    print("Done :)")

