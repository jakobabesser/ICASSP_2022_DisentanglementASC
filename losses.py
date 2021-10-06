import tensorflow as tf
import numpy as np

""" Helper functions for different losses used """

cce = tf.metrics.categorical_crossentropy
eps = 0.0000001
log2 = np.log(2.0)

def unmasked_cce(x, y):
    """ Categorical cross-entropy between input tensors
    Args:
        x: (tf.Tensor): Ground truth (batch_size x n_classes)
        y: (tf.Tensor): Prediction (batch_size x n_classes)
    Returns:
        cce (float): Categorical cross-entropy
    """
    return tf.math.reduce_mean(cce(x, y))


def masked_cce(x, mx, y, my):
    """ Categorical cross-entropy between subsets of input tensors
    Args:
        x: (tf.Tensor): Ground truth (batch_size x n_classes)
        mx: (tf.Tensor): Boolean mask to be applied on x to filter out subset (n_classes)
        y: (tf.Tensor): Prediction (batch_size x n_classes)
        my: (tf.Tensor): Boolean mask to be applied on y to filter out subset (n_classes)
    Returns:
        cce (float): Categorical cross-entropy
    """
    return unmasked_cce(tf.boolean_mask(x, mx),
                        tf.boolean_mask(y, my))


def unmasked_var(x):
    """ Batch-wise average over variance over predictions
    Args:
        x (tf.Tensor): Class prediction (batch_size x n_classes)
    Returns:
        mean_var (float): Batch-wise variance average
    """
    return tf.math.reduce_mean(tf.math.reduce_variance(x, axis=1))


def masked_var(x, m):
    """ Masked version of variance-based loss
    Args:
        x (tf.Tensor): Class prediction (batch_size x n_classes)
        m: (tf.Tensor): Boolean mask to be applied on x to filter out subset (n_classes)
    Returns:
        mean_var (float): Batch-wise variance average
    """
    return unmasked_var(tf.boolean_mask(x, m))


def loss_asc(do_masking: bool = False,
             use_td_targets: bool = True,
             asc_and_dc: bool = True,
             loss_weight_var: float = 500) -> float:
    """ Loss function for acoustic scene classification
    Args:
        do_masking (bool): if True, embedding masking is applied and embedding size is doubled
                           (as we effectively only use half of it then for learning ASC so that
                           this is comparable to the no-masking settings)
        use_td_targets (bool): Switch whether to use target domain annotations
        asc_and_dc (bool): Switch whether to generate training data for both ASC and domain
                           classification (True) or just ASC (False)
        loss_weight_var (float): Weighting factor of variance loss
    Returns:
        loss (float): Loss value
    """
    def loss(asc_true_mix, asc_pred):
        # in the generator (generator.py), we appended the source-domain mask (to identify which sample in the
        # current batch belongs to the source and target domain, respectively)
        asc_true = asc_true_mix[:, :-1]

        # two boolean masks to determine which samples belong to which domain
        mask_source = tf.cast(asc_true_mix[:, -1], bool)
        mask_target = tf.math.logical_not(mask_source)

        batch_size = asc_pred.shape[0]
        bs_half = int(batch_size/2)

        if do_masking:

            # relevant masks for loss functions
            # examples in first half of the batch
            mask_up = np.concatenate((np.ones(bs_half),
                                      np.zeros(bs_half))).astype(bool)
            mask_down = np.concatenate((np.zeros(bs_half),
                                        np.ones(bs_half))).astype(bool)

            # source domain examples in first half of the batch
            mask_source_up = tf.math.logical_and(mask_source, mask_up)

            if use_td_targets:
                # C3M
                cur_loss = masked_cce(asc_true, mask_up,
                                      asc_pred, mask_up) + \
                           loss_weight_var*masked_var(asc_pred,
                                                      mask_down)

            else:
                # C2M
                cur_loss = masked_cce(asc_true, mask_source_up,
                                      asc_pred, mask_source_up)
        else:
            if asc_and_dc:
                if use_td_targets:
                    # C3
                    cur_loss = unmasked_cce(asc_true, asc_pred)
                else:
                    # C2
                    cur_loss = masked_cce(asc_true, mask_source,
                                          asc_pred, mask_source)
            else:
                if use_td_targets:
                    # C1
                    cur_loss = masked_cce(asc_true, mask_target,
                                          asc_pred, mask_target) + \
                               masked_cce(asc_true, mask_source,
                                          asc_pred, mask_source)
                else:
                    # C0
                    cur_loss = masked_cce(asc_true, mask_source,
                                          asc_pred, mask_source)

        return cur_loss
    return loss


def loss_dc(do_masking: bool = False,
            use_td_targets: bool = True,
            asc_and_dc: bool = True,
            loss_weight_var: float = 500,
            loss_weight: float = 100) -> float:
    """ Loss function for domain classification
    Args:
        do_masking (bool): if True, embedding masking is applied and embedding size is doubled
                           (as we effectively only use half of it then for learning ASC so that
                           this is comparable to the no-masking settings)
        use_td_targets (bool): Switch whether to use target domain annotations
        asc_and_dc (bool): Switch whether to generate training data for both ASC and domain
                           classification (True) or just ASC (False)
        loss_weight_var (float): Weighting factor of variance loss
        loss_weight (float): Weighting factor for DC loss
    Returns:
        loss (float): Loss value
    """
    def loss(dc_true_mix, dc_pred):

        # in the generator (generator.py), we appended the source-domain mask (to identify which sample in the
        # current batch belongs to the source and target domain, respectively)
        dc_true = dc_true_mix[:, :-1]
        # mask_source = tf.cast(dc_true_mix[:, -1], bool)  # not needed here

        batch_size = dc_pred.shape[0]
        bs_half = int(batch_size/2)

        if do_masking:

            # relevant masks for loss functions
            # examples in first half of the batch
            mask_up = np.concatenate((np.ones(bs_half),
                                      np.zeros(bs_half))).astype(bool)
            # examples in the second half
            mask_down = np.concatenate((np.zeros(bs_half),
                                        np.ones(bs_half))).astype(bool)

            if use_td_targets:
                # C3M
                cur_loss = masked_cce(dc_true, mask_down,
                                      dc_pred, mask_down) + \
                           loss_weight_var*masked_var(dc_pred, mask_up)
            else:
                # C2M
                cur_loss = masked_cce(dc_true, mask_down,
                                      dc_pred, mask_down) + \
                           loss_weight_var*masked_var(dc_pred, mask_up)

        else:
            if asc_and_dc:
                if use_td_targets:
                    # C3
                    cur_loss = unmasked_cce(dc_true, dc_pred)
                    #loss_weight_var*unmasked_var(dc_pred)
                else:
                    # C2
                    cur_loss = unmasked_cce(dc_true, dc_pred)
                    # loss_weight_var*unmasked_var(dc_pred)
            else:
                # C0 / C1
                cur_loss = 0.0
        return loss_weight*cur_loss
    return loss


