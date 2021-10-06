import numpy as np


def mezza_normalization(feat_target, feat_source, eps=0.000001, switch_time_freq=False):
    """ Band-wise statistic matching between source and target domains
    Ref:
        Mezza, A. I., Habets, E. A. P., Müller, M., & Sarti, A. (2021). Unsupervised domain adaptation
        for acoustic scene classification using band-wise statistics matching. Proceedings of the European
        Signal Processing Conference (EUSIPCO), 11–15. https://doi.org/10.23919/Eusipco47968.2020.9287533
    """
    if switch_time_freq:
        # to be used if spectrogram is transposed in feature tensor
        feat_source = np.transpose(feat_source, (0, 2, 1))
        feat_target = np.transpose(feat_target, (0, 2, 1))

    ns, m, k = feat_source.shape
    nt, m, k = feat_target.shape

    # reshape to 2D by concatenating all frames of all batches
    x_s = np.reshape(feat_source, (ns*m, k))
    x_t = np.reshape(feat_target, (nt*m, k))

    # compute statistics over both domains
    mu_s = np.mean(x_s, axis=0)
    var_s = np.var(x_s, axis=0)
    mu_t = np.mean(x_t, axis=0)
    var_t = np.var(x_t, axis=0)

    # standardize target domain features (zero mean, unit variance)
    z_t = (x_t - mu_t)/(var_t+eps)

    # align target domain features to source domain distribution (mu_s, var_s)
    x_t_aligned = var_s*z_t + mu_s

    # reshape back to 3D (batch, time, freq)
    x_t_aligned = np.reshape(x_t_aligned, (nt, m, k))

    if switch_time_freq:
        # switch time & frequency again if necessary
        x_t_aligned = np.transpose(x_t_aligned, (0, 2, 1))

    return x_t_aligned

