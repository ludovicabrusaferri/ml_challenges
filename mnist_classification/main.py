# Copyright University College London, LSBU, 2023
# Author: Alexander C. Whitehead, Department of Computer Science, UCL
# Author: Ludovica Brusaferri, Department of Computer Science and Informatics, School of Engineering, LSBU
# For internal research only.

import random
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skimage.transform import rescale
from skimage.filters import gaussian, unsharp_mask
import einops
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt


def get_input(datasetname):
    print("get_input")

    split_ratio = 0.8
    dataset = tfds.load(datasetname)

    # Check if the dataset has both "train" and "test" splits
    if "train" in dataset and "test" in dataset:
        dataset_train = dataset["train"]
        dataset_validation = dataset["test"]
    else:
        # If not, split the dataset into training and validation sets
        dataset_size = tf.data.experimental.cardinality(dataset["train"]).numpy()
        num_train_examples = int(dataset_size * split_ratio)

        dataset_train = dataset["train"].take(num_train_examples)
        dataset_validation = dataset["train"].skip(num_train_examples)

    x_train = []
    x_test = []

    y_train = []
    y_test = []

    for example in dataset_train:
        x_train.append(example["image"].numpy().astype(np.float32))
        y_train.append(example["label"].numpy().astype(np.float32))

    for example in dataset_validation:

        x_test.append(example["image"].numpy().astype(np.float32))
        y_test.append(example["label"].numpy().astype(np.float32))

    # Assuming y_train and y_test are 1D arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


def get_next_geometric_value(an, a0):

    n = np.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2.0, (np.ceil(n) - 1.0))

    return an


def reshape_images_array(images):
    print("reshape_images")

    max_dimension_size = np.max(images.shape[1:-1])
    output_dimension_size = get_next_geometric_value(max_dimension_size, 2.0)

    rescaled_images = []

    for i in range(len(images)):
        rescaled_images.append(rescale(images[i], output_dimension_size / max_dimension_size, mode="constant",
                                       clip=False, preserve_range=True, channel_axis=-1))

    images = np.array(rescaled_images)

    while images.shape[1] + 1 < output_dimension_size:
        images = np.pad(images, ((0, 0), (1, 1), (0, 0), (0, 0)))  # noqa

    if images.shape[1] < output_dimension_size:
        images = np.pad(images, ((0, 0), (0, 1), (0, 0), (0, 0)))  # noqa

    while images.shape[2] + 1 < output_dimension_size:
        images = np.pad(images, ((0, 0,), (0, 0), (1, 1), (0, 0)))  # noqa

    if images.shape[2] < output_dimension_size:
        images = np.pad(images, ((0, 0), (0, 0), (0, 1), (0, 0)))  # noqa

    return images


def preprocess_images_array(x_train, x_test):
    print("preprocess_images")

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    standard_scaler = StandardScaler()

    x_train = np.reshape(standard_scaler.fit_transform(np.reshape(x_train, (-1, 1))), x_train.shape)
    x_test = np.reshape(standard_scaler.transform(np.reshape(x_test, (-1, 1))), x_test.shape)

    x_train = reshape_images_array(x_train)
    x_test = reshape_images_array(x_test)

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

    return x_train, x_test


def pad_image(image, output_dimension_size):
    while image.shape[0] + 1 < output_dimension_size:
        image = np.pad(image, ((1, 1), (0, 0), (0, 0)))  # noqa

    if image.shape[0] < output_dimension_size:
        image = np.pad(image, ((0, 1), (0, 0), (0, 0)))  # noqa

    while image.shape[1] + 1 < output_dimension_size:
        image = np.pad(image, ((0, 0), (1, 1), (0, 0)))  # noqa

    if image.shape[1] < output_dimension_size:
        image = np.pad(image, ((0, 0), (0, 1), (0, 0)))  # noqa

    return image


def reshape_images_list(images):
    print("reshape_images")

    max_dimension_size = -1

    for i in range(len(images)):
        current_max_dimension_size = np.max(images[i].shape[:-1])

        if current_max_dimension_size > max_dimension_size:
            max_dimension_size = current_max_dimension_size

    output_dimension_size = get_next_geometric_value(max_dimension_size, 2.0)

    reshaped_images = []

    for i in range(len(images)):
        image = rescale(images[i], output_dimension_size / np.max(images[i].shape[:-1]), mode="constant", clip=False,
                        preserve_range=True, channel_axis=-1)

        image = pad_image(image, output_dimension_size)

        reshaped_images.append(image)

    images = np.array(reshaped_images)

    return images


def preprocess_images_list(x_train, x_test):
    print("preprocess_images")

    standard_scaler = StandardScaler()

    x_train_len = len(x_train)

    for i in range(x_train_len):
        standard_scaler.partial_fit(np.reshape(x_train[i], (-1, 1)))

    for i in range(x_train_len):
        x_train[i] = np.reshape(standard_scaler.transform(np.reshape(x_train[i], (-1, 1))), x_train[i].shape)

    for i in range(len(x_test)):
        x_test[i] = np.reshape(standard_scaler.transform(np.reshape(x_test[i], (-1, 1))), x_test[i].shape)

    x_train = reshape_images_list(x_train)
    x_test = reshape_images_list(x_test)

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

    return x_train, x_test


def preprocess_labels(y_train, y_test):
    print("preprocess_labels")

    one_hot_encoder = OneHotEncoder(sparse_output=False)

    y_train = np.reshape(one_hot_encoder.fit_transform(np.reshape(y_train, (-1, 1))),
                         (y_train.shape[0], -1))
    y_test = np.reshape(one_hot_encoder.transform(np.reshape(y_test, (-1, 1))),
                        (y_test.shape[0], -1))

    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    return y_train, y_test


def preprocess_input(x_train, x_test, y_train, y_test):
    print("preprocess_input")

    x_train, x_test = preprocess_images_array(x_train, x_test)
    y_train, y_test = preprocess_labels(y_train, y_test)

    return x_train, x_test, y_train, y_test


def get_model_dense(x_train, y_train):
    print("get_model")

    x_input = tf.keras.Input(shape=x_train.shape[1:])
    x = x_input

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=x.shape[-1])(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=y_train.shape[1])(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def leaky_relu(x):
    x = tf.math.maximum(0.1 * x, x)
    return x


def conv_block(x, filters, kernel_size, activation, num_conv):
    for i in range(num_conv):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size)(x)
        x = tf.keras.layers.Lambda(activation)(x)
    return x


def down_block(x, filters, kernel_size, strides, activation):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = tf.keras.layers.Lambda(activation)(x)
    return x

def dense_block(x, units, activation):
    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.Lambda(activation)(x)

    return x


def get_model_ludo(x_train, y_train, input_kernel_size):
    print("get_model")

    x_input = tf.keras.Input(shape=x_train.shape[1:])
    x = x_input

    filters = [64, 128]  # standard to start with 128

    x = conv_block(x, filters=filters[0], kernel_size=input_kernel_size, activation=leaky_relu, num_conv=1)

    # leaky relu allows negative updates

    x = conv_block(x, filters=filters[0], kernel_size=(3, 3), activation=leaky_relu, num_conv=2)

    x = down_block(x, filters=filters[0], kernel_size=(3, 3), strides=(2, 2), activation=leaky_relu)

    x = conv_block(x, filters=filters[1], kernel_size=(3, 3), activation=leaky_relu, num_conv=2)

    # Flatten
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    # Dense layers
    x = dense_block(x, units=x.shape[-1], activation=leaky_relu)

    # Dense layers
    x = dense_block(x, units=y_train.shape[1], activation=tf.keras.activations.softmax)

    model = tf.keras.Model(inputs=[x_input], outputs=[x])

    return model


def get_model_conv(x_train, y_train, input_kernel_size, strides):
    print("get_model")

    start_power = 5
    num_elements = 4  # Change this to the desired number
    filters = [pow(2, i) for i in range(start_power, start_power + num_elements)]

    x_input = tf.keras.Input(shape=x_train.shape[1:])
    x = x_input

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(input_kernel_size, input_kernel_size),
                               strides=(strides, strides),
                               padding="same",
                               dilation_rate=1,
                               kernel_initializer=tf.keras.initializers.he_uniform)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    for i in range(len(filters)):
        x_res = x

        for j in range(2):
            x = tf.keras.layers.GroupNormalization(groups=1)(x)
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       dilation_rate=1,
                                       kernel_initializer=tf.keras.initializers.he_uniform)(x)
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.GroupNormalization(groups=1)(x_res)
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       dilation_rate=1,
                                       kernel_initializer=tf.keras.initializers.he_uniform)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x = tf.keras.layers.GroupNormalization(groups=1)(x)
        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   dilation_rate=1,
                                   kernel_initializer=tf.keras.initializers.he_uniform)(x)
        x = tf.keras.layers.Lambda(einops.rearrange,
                                   arguments={"pattern": "b (h h1) (w w1) c -> b h w (c h1 w1)", "h1": 2, "w1": 2})(x)

    for j in range(2):
        x = tf.keras.layers.GroupNormalization(groups=1)(x)
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   dilation_rate=1,
                                   kernel_initializer=tf.keras.initializers.he_uniform)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.GroupNormalization(groups=1)(x)
    x = tf.keras.layers.Dense(units=x.shape[-1],
                              kernel_initializer=tf.keras.initializers.he_uniform)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.GroupNormalization(groups=1)(x)
    x = tf.keras.layers.Dense(units=y_train.shape[1],
                              kernel_initializer=tf.keras.initializers.zeros)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def sinespace(start, stop, num):
    linspaced = np.linspace(0.0, 90.0, num)

    sinspaced = np.sin(np.deg2rad(linspaced))

    sinspaced_min = np.min(sinspaced)
    sinspaced = start + ((sinspaced - sinspaced_min) * ((stop - start) / ((np.max(sinspaced) - sinspaced_min) +
                                                                          np.finfo(np.float32).eps)))

    return sinspaced


def flip_image(image):
    if random.choice([True, False]):
        image = np.flip(image, axis=0)

    if random.choice([True, False]):
        image = np.flip(image, axis=1)

    return image


def scale_image(image):
    scale = random.uniform(1.0, 1.5)

    if random.choice([True, False]):
        scale = 1.0 / scale

    image = rescale(image, scale, mode="constant", clip=False, preserve_range=True, channel_axis=-1)

    return image


def translate_image(image):
    max_translation = int(np.round(image.shape[0] / 4.0))

    translation = random.randint(0, max_translation)

    if random.choice([True, False]):
        image = np.pad(image, ((0, translation), (0, 0), (0, 0)))  # noqa
    else:
        image = np.pad(image, ((translation, 0), (0, 0), (0, 0)))  # noqa

    translation = random.randint(0, max_translation)

    if random.choice([True, False]):
        image = np.pad(image, ((0, 0), (0, translation), (0, 0)))  # noqa
    else:
        image = np.pad(image, ((0, 0), (translation, 0), (0, 0)))  # noqa

    return image


def crop_image(image, output_dimension_size):
    while image.shape[0] - 1 > output_dimension_size:
        image = image[1:-1]

    if image.shape[0] > output_dimension_size:
        image = image[1:]

    while image.shape[1] - 1 > output_dimension_size:
        image = image[:, 1:-1]

    if image.shape[1] > output_dimension_size:
        image = image[:, 1:]

    return image


def augmentation(image):
    image = image.numpy()

    input_dimension_size = image.shape[0]

    # image = flip_image(image)

    image = gaussian(image, sigma=random.uniform(0.0, 1.0), mode="constant", preserve_range=True, channel_axis=-1)
    image = unsharp_mask(image, radius=random.uniform(0.0, 1.0), amount=random.uniform(0.0, 1.0),
                         preserve_range=True, channel_axis=1)

    image = scale_image(image)
    image = scipy.ndimage.rotate(image, angle=random.uniform(-90.0, 90.0), axes=(0, 1), order=1)
    image = translate_image(image)

    image = pad_image(image, input_dimension_size)
    image = crop_image(image, input_dimension_size)

    image = tf.convert_to_tensor(image)

    return image


def get_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    loss = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
    loss = tf.math.reduce_mean(loss)

    return loss


def get_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    accuracy = tf.math.reduce_mean(accuracy)
    accuracy = accuracy * 100.0

    return accuracy


def train(model, optimiser, x_train, y_train):
    print("train")

    epochs = 8
    min_batch_size = 32
    max_batch_size = 32

    min_batch_size = get_next_geometric_value(min_batch_size, 2.0)
    batch_sizes = [min_batch_size]

    while True:
        current_batch_size = int(np.round(batch_sizes[-1] * 2.0))

        if current_batch_size <= max_batch_size:
            batch_sizes.append(current_batch_size)
        else:
            break

    del current_batch_size

    batch_sizes_epochs = sinespace(0.0, epochs - 1, len(batch_sizes) + 1)
    batch_sizes_epochs = np.round(batch_sizes_epochs)

    batch_sizes_epochs_len = len(batch_sizes_epochs)

    current_batch_size = None
    indices = list(range(x_train.shape[0]))

    for i in range(epochs):
        for j in range(batch_sizes_epochs_len - 1, 0, -1):
            if batch_sizes_epochs[j - 1] <= i <= batch_sizes_epochs[j]:
                current_batch_size = batch_sizes[j - 1]

                break

        iterations = int(np.floor(x_train.shape[0] / current_batch_size))

        np.random.shuffle(indices)

        for j in range(iterations):
            current_index = current_batch_size * j

            current_x_train = []
            y_true = []

            for k in range(current_batch_size):
                # augmented_x_train = augmentation(x_train[current_index])
                augmented_x_train = x_train[current_index]

                current_x_train.append(augmented_x_train)
                y_true.append(y_train[current_index])

                current_index = current_index + 1

            current_x_train = tf.convert_to_tensor(current_x_train)
            y_true = tf.convert_to_tensor(y_true)

            with tf.GradientTape() as tape:
                y_pred = model([current_x_train], training=True)

                loss = get_loss(y_true, y_pred)

            gradients = tape.gradient(loss, model.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            accuracy = get_accuracy(y_true, y_pred)

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss: {5:12} Accuracy: {6:6}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), str(loss.numpy()),
                str(accuracy.numpy())))

    return model


def train_gradient_accumulation(model, optimiser, x_train, y_train):
    print("train")

    y_train = tf.expand_dims(y_train, axis=1)

    epochs = 8
    min_batch_size = 32
    max_batch_size = x_train.shape[0]

    min_batch_size = get_next_geometric_value(min_batch_size, 2.0)
    batch_sizes = [min_batch_size]

    while True:
        current_batch_size = int(np.round(batch_sizes[-1] * 2.0))

        if current_batch_size <= max_batch_size:
            batch_sizes.append(current_batch_size)
        else:
            break

    del current_batch_size

    batch_sizes_epochs = sinespace(0.0, epochs - 1, len(batch_sizes) + 1)
    batch_sizes_epochs = np.round(batch_sizes_epochs)

    batch_sizes_epochs_len = len(batch_sizes_epochs)

    current_batch_size = None
    indices = list(range(x_train.shape[0]))

    for i in range(epochs):
        for j in range(batch_sizes_epochs_len - 1, 0, -1):
            if batch_sizes_epochs[j - 1] <= i < batch_sizes_epochs[j]:
                current_batch_size = batch_sizes[j - 1]

                break

        iterations = int(np.floor(x_train.shape[0] / current_batch_size))

        np.random.shuffle(indices)

        for j in range(iterations):
            accumulated_gradients = [tf.zeros_like(trainable_variable) for trainable_variable in
                                     model.trainable_variables]

            current_index = current_batch_size * j

            losses = []
            accuracies = []

            for m in range(current_batch_size):
                #current_x_train = augmentation(x_train[current_index])
                current_x_train = x_train[current_index]

                y_true = y_train[current_index]

                current_x_train = tf.expand_dims(current_x_train, axis=0)

                with tf.GradientTape() as tape:
                    y_pred = model([current_x_train], training=True)

                    loss = get_loss(y_true, y_pred)

                gradients = tape.gradient(loss, model.trainable_weights)

                accumulated_gradients = [(accumulated_gradient + gradient) for accumulated_gradient, gradient in
                                         zip(accumulated_gradients, gradients)]

                losses.append(loss)
                accuracies.append(get_accuracy(y_true, y_pred))

                current_index = current_index + 1

            gradients = [gradient / current_batch_size for gradient in accumulated_gradients]
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            loss = tf.math.reduce_mean(losses)
            accuracy = tf.math.reduce_mean(accuracies)

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss: {5:12} Accuracy: {6:6}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), str(loss.numpy()),
                str(accuracy.numpy())))

    return model


def test(model, x_test, y_test):
    print("test")

    batch_size = 1024

    losses = []
    accuracies = []

    for i in range(int(np.floor(x_test.shape[0] / batch_size))):
        current_x_test = []
        y_true = []

        for j in range(batch_size):
            current_index = (i * batch_size) + j

            current_x_test.append(x_test[current_index])
            y_true.append(y_test[current_index])

        current_x_test = tf.convert_to_tensor(current_x_test)
        y_true = tf.convert_to_tensor(y_true)

        y_pred = model([current_x_test], training=True)

        losses.append(get_loss(y_true, y_pred))
        accuracies.append(get_accuracy(y_true, y_pred))

    loss = tf.math.reduce_mean(losses)
    accuracy = tf.math.reduce_mean(accuracies)

    print("Loss: {0:12} Accuracy: {1:6}%".format(str(loss.numpy()), str(accuracy.numpy())))

    return model, y_pred


def plot_test(x_test, y_test, preds):
    # Ensure x_test and preds have the same length
    if len(x_test) != len(preds):
        raise ValueError("x_test and preds must have the same length")

    # Create a 5x5 grid of subplots to display the images and predictions
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed

    for i in range(25):  # Display up to 25 images
        plt.subplot(5, 5, i + 1)  # Subplot index starts from 1
        plt.imshow(x_test[i], cmap='gray')
        plt.title("P: {0}, T: {1}".format(str(tf.argmax(preds[i]).numpy()), str(tf.argmax(y_test[i]).numpy())))
        plt.axis("off")

    plt.show()  # Display the entire grid of images


def main():
    print("main")

    x_train, x_test, y_train, y_test = get_input("cifar10") #mnist, cifar10
    x_train, x_test, y_train, y_test = preprocess_input(x_train, x_test, y_train, y_test)

    model = get_model_ludo(x_train, y_train, input_kernel_size=7)

    model.summary()

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-04,
                                         weight_decay=0)

    # train standard
    model = train(model, optimiser, x_train, y_train)

    _, y_pred = test(model, x_test, y_test)

    plot_test(x_test, y_test, y_pred)

    return True


if __name__ == "__main__":
    main()
