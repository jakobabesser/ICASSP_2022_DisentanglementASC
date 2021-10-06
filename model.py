from model_kaggle import create_model_kaggle
from losses import *


def create_core_model(shape_in: tuple, n_output_dim: int, emb_dim: int = 256) -> (tf.keras.models.Model, str, str):
    """ Create core model f()
    Args:
        shape_in (tuple): Input dimensions (excluding batch dimension)
        n_output_dim (int): Number of output classes
        emb_dim (int): Embedding dimension size
    Returns:
        model (tf.keras.models.Model): Core model
        first_layer_after_input_label (str): Layer name of first layer after the input layer
        emb_layer_label (str): Layer name of last layer before final classification layer
    """
    # initialize Kaggle model
    model = create_model_kaggle(shape_in=shape_in, num_output_dim=n_output_dim, emb_dim=emb_dim)

    # input layer label
    first_layer_after_input_label = model.layers[1].name

    # label of pre-final dense layer (will be considered as embedding layer)
    emb_layer_label = model.layers[-3].name

    print("Create core model starting from layer {} to layer {} of the Kaggle model".format(first_layer_after_input_label,
                                                                                            emb_layer_label))

    return model, first_layer_after_input_label, emb_layer_label


def create_model(shape_in: tuple,
                 n_output_dim: int,
                 do_masking: bool = False,
                 emb_dim: int = 256,
                 inference: bool = False) -> tf.keras.models.Model:
    """ Create model for disentanglement learning with embedding masking and dual classification tasks
    Args:
        shape_in (tuple): Input feature shape (4D)
        n_output_dim (int): Number of output classes
        do_masking (bool): Switch whether to double the embedding layer size (used if embedding masking is
                           applied)
        emb_dim (int): Embedding size
        inference (bool): Switch, whether model should be created for inference mode (True), where batch size must not be
                          specified in the Input Layer or not (False)
    Returns:
        disent_model (tf.keras.models.Model): Core model
    """
    # double embedding size if embedding masking is used
    if do_masking:
        emb_dim *= 2

    # create core model
    core_model, first_layer_after_input_label, emb_layer_label = create_core_model(shape_in[1:], n_output_dim,
                                                                                   emb_dim)

    # generate final model with two input branches (spectrogram patch & embedding mask)
    if inference:
        spec_in = tf.keras.Input(shape=shape_in[1:])
        emb_mask_in = tf.keras.Input(shape=(emb_dim,))
    else:
        # define batch-size during training
        spec_in = tf.keras.Input(shape=shape_in[1:], batch_size=shape_in[0])
        emb_mask_in = tf.keras.Input(shape=(emb_dim,), batch_size=shape_in[0])

    # integrate part of the core model (just until embedding vector)
    intermediate_model = tf.keras.models.Model(inputs=core_model.get_layer(first_layer_after_input_label).input,
                                               outputs=core_model.get_layer(emb_layer_label).get_output_at(0),
                                               name="core")
    emb = intermediate_model(spec_in)

    # normalize each embedding
    emb = tf.keras.layers.LayerNormalization(name="emb")(emb)

    # embedding masking
    x_m = tf.keras.layers.multiply((emb, emb_mask_in))

    # create two output layers for acoustic scene prediction (10 classes) and domain classification (3 classes)
    out_asc = tf.keras.layers.Dense(10, activation='softmax', name='asc')(x_m)
    out_dc = tf.keras.layers.Dense(3, activation='softmax', name='dc')(x_m)

    # finally connect to dual-input-dual-output model
    disent_model = tf.keras.models.Model(inputs=[spec_in, emb_mask_in], outputs=[out_asc, out_dc])

    return disent_model
