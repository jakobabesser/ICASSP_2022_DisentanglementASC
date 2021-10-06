import tensorflow.keras.layers as tkl
from tensorflow.keras.models import Model


def create_model_kaggle(shape_in,
                        num_output_dim,
                        emb_dim=256):
    """ "Kaggle" model for acoustic scene classification used in

    Mezza, A. I., Habets, E. A. P., Müller, M., & Sarti, A. (2021).
    #Unsupervised domain adaptation for acoustic scene classification
    using band-wise statistics matching. Proceedings of the European
    Signal Processing Conference (EUSIPCO), 11–15.
    https://doi.org/10.23919/Eusipco47968.2020.9287533"

    and before in

    Gharib, S., Drossos, K., Emre, C., Serdyuk, D., & Virtanen, T. (2018). Unsupervised Adversarial Domain
    Adaptation for Acoustic Scene Classification. Proceedings of the Detection and Classification of
    Acoustic Scenes and Events (DCASE). Surrey, UK.

    Drossos, K., Magron, P., & Virtanen, T. (2019). Unsupervised Adversarial Domain Adaptation based
    on the Wasserstein Distance for Acoustic Scene Classification. Proceedings of the IEEE Workshop
    on Applications of Signal Processing to Audio and Acoustics (WASPAA), 259–263. New Paltz, NY, USA.

    """
    inp = tkl.Input(shape_in)
    x = tkl.ZeroPadding2D(5)(inp)
    x = tkl.Conv2D(48, kernel_size=(11, 11), padding='valid', strides=(2, 3))(x)
    x = tkl.Activation(activation="relu")(x)
    x = tkl.MaxPooling2D(pool_size=(3, 3), strides=(1, 2))(x)
    x = tkl.BatchNormalization()(x)

    x = tkl.ZeroPadding2D(2)(x)
    x = tkl.Conv2D(128, kernel_size=(5, 5), padding='valid', strides=(2, 3))(x)
    x = tkl.Activation(activation="relu")(x)
    x = tkl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tkl.BatchNormalization()(x)

    x = tkl.ZeroPadding2D(1)(x)
    x = tkl.Conv2D(192, kernel_size=(3, 3), padding='valid', strides=(1, 1))(x)
    x = tkl.Activation(activation="relu")(x)
    x = tkl.ZeroPadding2D(1)(x)
    x = tkl.Conv2D(192, kernel_size=(3, 3), padding='valid', strides=(1, 1))(x)
    x = tkl.Activation(activation="relu")(x)

    x = tkl.Conv2D(128, kernel_size=(3, 3), padding='valid', strides=(1, 1))(x)
    x = tkl.Activation(activation="relu")(x)
    x = tkl.MaxPooling2D(pool_size=(3, 3), strides=(1, 2))(x)
    x = tkl.BatchNormalization()(x)

    x = tkl.Flatten()(x)

    x = tkl.Dense(emb_dim)(x)
    x = tkl.Activation(activation="relu")(x)

    # note we do not apply dropout since we avoid data augmentation / regularization in our experiments to better
    # evaluate the effectiveness of the disentanglement learning approach

    x = tkl.Dense(emb_dim)(x)
    x = tkl.Activation(activation="relu")(x)

    x = tkl.Dense(num_output_dim, activation="softmax")(x)

    model = Model(inputs=inp, outputs=x)

    return model

