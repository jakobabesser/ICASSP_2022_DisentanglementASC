import numpy as np
import pickle
import torch
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import os
from normalization import mezza_normalization

class_labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian',
                'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']

# mapping functions between class ids and labels
id_to_class_lab = {_: class_labels[_] for _ in range(10)}
class_lab_to_id = {class_labels[_]: _ for _ in range(10)}


def get_array_from_torch_data(data, lab: str, is_feat: bool = True) -> np.ndarray:
    """ Decode numpy array from PyTorch data as published at https://zenodo.org/record/1401995
    Args:
        data (Torch data): data stored in pickle files
        lab (string): Label of subset to be extracted ("A", "B", or "C")
        is_feat (bool): There are pickle files for feature arrays (4D) and target arrays (2D),
                        if "True", we need to re-arrange the Torch data order (batch, channel, x, y)
                        to the Keras / TF order (batch, x, y, channel),
                        if "False" (target case), we can just use the data as it is
    """
    f = data[lab].cpu().detach().numpy()
    s = f.shape
    if is_feat:
        f = f.reshape((s[0], s[2], s[3], s[1]))
    return f


def load_data_from_pickle_file(fn_pickle: str):
    """ Import raw data from pickle file
    Args:
        fn_pickle (str): Pickle file name
    """
    with open(fn_pickle, 'rb') as f:
        data = pickle.load(f)
    return data


class DataGenerator(Sequence):
    """ Data generator that implements embedding masking for disentanglement learning for ASC """

    def __init__(self,
                 ds_type: str,
                 dir_feat: str,
                 batch_size: int = 512,
                 do_masking: bool = False,
                 use_td: bool = True,
                 asc_and_dc: bool = True,
                 n_batches_per_epoch: int = 500,
                 n_dc_classes: int = 3,
                 emb_dim: int = 256):
        """ Initialize class
        Args:
            ds_type (str): Dataset type (training or test)
            dir_feat (str): Directory where pickle files (from https://zenodo.org/record/1401995) are located
            batch_size (int): Batch size
            do_masking (bool): if True, embedding masking is applied and embedding size is doubled
                               (as we effectively only use half of it then for learning ASC so that
                               this is comparable to the no-masking settings)
            use_td (bool): Switch whether to use target domain data (or just source domain data)
                           for training
            asc_and_dc (bool): Switch whether to generate training data for both ASC and domain
                               classification (True) or just ASC (False)
            n_batches_per_epoch (int): Number of batches per epoch
            n_dc_classes (int): Number of domain classification classes
            emb_dim (int): Embedding size
        """
        assert ds_type in ('training', 'test')  # we don't use the validation set
        self.type = ds_type
        self.dir_feat = dir_feat
        self.batch_size = batch_size
        self.half_batch_size = batch_size//2
        self.do_masking = do_masking
        self.use_td = use_td
        self.asc_and_dc = asc_and_dc
        self.n_batches_per_epoch = n_batches_per_epoch
        self.emb_dim = emb_dim
        self.n_dc_classes = n_dc_classes
        self.device_labels = ('A', 'B', 'C')

        # double embedding size if masking is applied (C2M and C3M)
        if do_masking:
            self.emb_dim *= 2
        self.half_emb_dim = self.emb_dim // 2

        print('Load data in generator ')
        feat_subsets = []
        asc_target_subsets = []
        dc_target_subsets = []
        fn_feat = os.path.join(dir_feat, '{}_features.p'.format(self.type))
        fn_asc_target = os.path.join(dir_feat, '{}_scene_labels.p'.format(self.type))

        # load raw data and ASC targets
        feat_data = load_data_from_pickle_file(fn_feat)
        asc_target_data = load_data_from_pickle_file(fn_asc_target)

        # iterate over device-subsets ("A", "B", and "C") in the dataset and concatenate all of them
        for i, dl in enumerate(self.device_labels):
            feat_subsets.append(get_array_from_torch_data(feat_data, dl, is_feat=True))
            asc_target_subsets.append(get_array_from_torch_data(asc_target_data, dl, is_feat=False))

            # create device targets (one-hot encoding)
            n_patches = feat_subsets[-1].shape[0]
            curr_dt = np.zeros((n_patches, self.n_dc_classes), dtype=np.int8)
            curr_dt[:, i] = 1
            dc_target_subsets.append(curr_dt)

        self.feat = np.concatenate(feat_subsets)
        self.asc_target = np.concatenate(asc_target_subsets)
        self.dc_target = np.concatenate(dc_target_subsets)

        # derive indices of source and target domain examples
        dc_idx = np.argmax(self.dc_target, axis=1)
        self.idx_source_domain = np.where(dc_idx == 0)[0]
        self.idx_target_domain = np.where(dc_idx > 0)[0]

    def __len__(self):
        """ Return number of batches per epoch"""
        return self.n_batches_per_epoch

    def __getitem__(self, index: int):
        """ Return one training mini-batch """

        # shuffle source and target domain indices for random sampling
        np.random.shuffle(self.idx_source_domain)
        np.random.shuffle(self.idx_target_domain)

        batch_target_dc = np.zeros((self.batch_size, self.n_dc_classes), dtype=np.int8)

        # in case we include target domain data
        if self.use_td:

            # get random sample indices for source and target domain data to an equal amount
            cur_idx_source_domain = self.idx_source_domain[:self.half_batch_size]
            cur_idx_target_domain = self.idx_target_domain[:self.half_batch_size]

            # collect all indices for this batch
            cur_idx = np.concatenate((cur_idx_source_domain, cur_idx_target_domain))

            # randomly shuffle source and target domain examples across the mini-batch
            mix_idx = np.arange(self.batch_size)
            np.random.shuffle(mix_idx)
            cur_idx = cur_idx[mix_idx]

            # prepare features & ASC & DC targets for this batch
            batch_feat = self.feat[cur_idx, :, :, :]
            batch_target_asc = self.asc_target[cur_idx, :]
            batch_target_dc = self.dc_target[cur_idx, :]

        # in case we just use source domain data
        else:

            # get random sample indices for source domain (only)
            cur_idx_source_domain = self.idx_source_domain[:self.batch_size]

            # prepare features & ASC / DC targets
            batch_feat = self.feat[cur_idx_source_domain, :, :, :]
            batch_target_asc = self.asc_target[cur_idx_source_domain, :]
            batch_target_dc[:, 0] = 1  # not used in this case

        # generate embedding mask
        batch_mask = np.zeros((self.batch_size, self.emb_dim), dtype=np.float16)

        # if masking is selected, we create a symmetrical mask where the first half of the embedding
        # dimensions is "on" for the first half of the batch data and the second half of the
        # dimensions is on for the second half, the others are set to zero
        if self.do_masking:
            batch_mask[:self.half_batch_size, :self.half_emb_dim] = 1
            batch_mask[self.half_batch_size:, self.half_emb_dim:] = 1
        else:
            # all ones if masking is not active (so that the multiplication in the model has no effect)
            batch_mask[:, :] = 1

        # append one column with source domain mask:
        #  this is split again in the loss functions, this way we can forward the information
        #  whether a sample in the batch belongs to the source or target domain to the loss function)
        batch_target_asc = np.hstack((batch_target_asc, batch_target_dc[:, 0][:, np.newaxis]))
        batch_target_dc = np.hstack((batch_target_dc, batch_target_dc[:, 0][:, np.newaxis]))

        # return data for dual-input & dual-output model
        return (batch_feat, batch_mask), (batch_target_asc, batch_target_dc)

    def on_epoch_end(self):
        pass
