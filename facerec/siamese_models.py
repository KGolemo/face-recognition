import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inceptionv3_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50_input
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, Lambda, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History

from datetime import datetime
from facerec.utils import euclidean_distance, contrastive_loss, accuracy
from facerec.dataset_generator import DataGenerator

from typing import Tuple


def get_base_model(base_architecture: str = 'ResNet50',
                   input_shape: Tuple[int, int, int] = (150, 150, 3)) -> Sequential:
    """
    Get a pre-trained base model for a Siamese network.

    Parameters
    ----------
    base_architecture : str, optional
        The architecture of the pre-trained model (default: 'ResNet50').
    input_shape : tuple[int, int, int], optional
        The input shape for the base model (default: (150, 150, 3)).

    Returns
    -------
    Sequential
        Pre-trained base model.

    Raises
    ------
    ValueError
        If an unsupported base architecture is provided.

    Example
    -------
    >>> base_model = get_base_model('VGG16', (224, 224, 3))
    """
    if base_architecture == 'ResNet50':
        pretrained_model = ResNet50(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False)
    elif base_architecture == 'VGG16':
        pretrained_model = VGG16(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False)
    elif base_architecture == 'InceptionV3':
        pretrained_model = InceptionV3(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False)
    else:
        raise ValueError("Invalid base architecture. Supported architectures are 'ResNet50', 'VGG16', and 'InceptionV3'.")

    # Freeze the base model
    pretrained_model.trainable = False

    # Use a Sequential model to add a trainable classifier on top
    base_network = Sequential([
        pretrained_model,
        Flatten(),
        Dense(64, activation='relu')],
        name=f'Pretrained_{base_architecture}')

    base_network.summary()

    return base_network


def get_siamese_model(base_architecture: str = 'ResNet50',
                      input_shape: Tuple[int, int, int] = (150, 150, 3)) -> Model:
    """
    Create a Siamese network model.

    Parameters
    ----------
    base_architecture : str, optional
        The architecture of the pre-trained base model (default: 'ResNet50').
    input_shape : tuple[int, int, int], optional
        The input shape for the model (default: (150, 150, 3)).

    Returns
    -------
    Model
        Siamese network model.

    Raises
    ------
    ValueError
        If an unsupported base architecture is provided.

    Example
    -------
    >>> siamese_model = get_siamese_model('VGG16', (224, 224, 3))
    """
    base_network = get_base_model(base_architecture, input_shape)

    left_input = Input(input_shape, name='input_1')
    right_input = Input(input_shape, name='input_2')

    # Apply architecture-specific preprocessing
    if base_architecture == 'ResNet50':
        preprocess_fn = preprocess_resnet50_input
    elif base_architecture == 'VGG16':
        preprocess_fn = preprocess_vgg16_input
    elif base_architecture == 'InceptionV3':
        preprocess_fn = preprocess_inceptionv3_input
    else:
        raise ValueError("Invalid base architecture. Supported architectures are 'ResNet50', 'VGG16', and 'InceptionV3'.")

    left_input_preprocessed = preprocess_fn(left_input)
    right_input_preprocessed = preprocess_fn(right_input)

    # Since this is a siamese nn, both sides share the same network.
    encoded_l = base_network(left_input_preprocessed)
    encoded_r = base_network(right_input_preprocessed)

    # The euclidean distance layer outputs close to 0 value when two inputs are similar and 1 otherwise.
    distance = Lambda(euclidean_distance, name='euclidean_distance')([encoded_l, encoded_r])

    siamese_network = Model(inputs=[left_input, right_input], outputs=distance)

    siamese_network.compile(loss=contrastive_loss, optimizer=Adam(0.0001, beta_1=0.99), metrics=[accuracy])

    return siamese_network


def train_model(siamese_network: Model,
                model_name: str,
                training_dataset_generator: DataGenerator,
                validation_dataset_generator: DataGenerator,
                num_epochs: int = 50) -> History:
    """
    Train a Siamese network model.

    Parameters
    ----------
    siamese_network : Model
        The Siamese network model to train.
    model_name : str
        Name of the model for saving checkpoints and logs.
    training_dataset_generator : DataGenerator
        The generator for the training dataset.
    validation_dataset_generator : DataGenerator
        The generator for the validation dataset.
    num_epochs : int, optional
        Number of training epochs (default: 50).

    Returns
    -------
    History
        Training history.

    Example
    -------
    >>> history = train_model(siamese_model, 'my_siamese_model', training_data_gen, validation_data_gen)
    """
    checkpoint_dir = os.path.join(r'../checkpoints/models/', model_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=0)

    log_dir = os.path.join(r'../logs/models/', model_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

    history = siamese_network.fit(
        training_dataset_generator,
        epochs=num_epochs,
        validation_data=validation_dataset_generator,
        callbacks=[es, checkpoint, tensorboard_callback],
        verbose=1,
        # workers=4,
        # use_multiprocessing=True,
        max_queue_size=32
    )

    return history
