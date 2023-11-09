# Copyright University College London, LSBU, 2023
# Author: Alexander C. Whitehead, Department of Computer Science, UCL
# Author: Ludovica Brusaferri, Department of Computer Science and Informatics, School of Engineering, LSBU
# For internal research only.

import numpy as np
import einops
import tensorflow_datasets as tfds
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skimage.transform import rescale
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
        x_train.append(example["image"].numpy())
        y_train.append(example["label"].numpy())

    for example in dataset_validation:
        x_test.append(example["image"].numpy())
        y_test.append(example["label"].numpy())

    # Convert to NumPy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # Assuming y_train and y_test are 1D arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


def reshape_images(images):
    print("reshape_images")

    output_dimension_size = 32

    scaled_images = []

    for i in range(images.shape[0]):
        scaled_images.append(rescale(images[i], output_dimension_size / np.max(images.shape[1:-1]), mode="constant",
                                     clip=False, channel_axis=-1))

    images = np.array(scaled_images)

    while images.shape[1] + 1 < output_dimension_size:
        images = np.pad(images, ((0, 0), (1, 1), (0, 0), (0, 0)))  # noqa

    if images.shape[1] < output_dimension_size:
        images = np.pad(images, ((0, 0), (1, 1), (0, 0), (0, 0)))  # noqa

    while images.shape[2] + 1 < output_dimension_size:
        images = np.pad(images, ((0, 0), (0, 0), (1, 1), (0, 0)))  # noqa

    if images.shape[2] < output_dimension_size:
        images = np.pad(images, ((0, 0), (0, 0), (0, 1), (0, 0)))  # noqa

    return images


def preprocess_images(x_train, x_test):
    print("preprocess_images")

    standard_scaler = StandardScaler()

    x_train = np.reshape(standard_scaler.fit_transform(np.reshape(x_train, (-1, 1))), x_train.shape)
    x_test = np.reshape(standard_scaler.transform(np.reshape(x_test, (-1, 1))), x_test.shape)

    x_train = reshape_images(x_train)
    x_test = reshape_images(x_test)

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)

    return x_train, x_test


def preprocess_labels(y_train, y_test):
    print("preprocess_labels")

    one_hot_encoder = OneHotEncoder(sparse_output=False)

    y_train = np.reshape(one_hot_encoder.fit_transform(np.reshape(y_train, (-1, 1))),
                         (y_train.shape[0], -1))
    y_test = np.reshape(one_hot_encoder.transform(np.reshape(y_test, (-1, 1))),
                        (y_test.shape[0], -1))

    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    return y_train, y_test


def preprocess_input(x_train, x_test, y_train, y_test):
    print("preprocess_input")

    x_train, x_test = preprocess_images(x_train, x_test)
    y_train, y_test = preprocess_labels(y_train, y_test)

    return x_train, x_test, y_train, y_test


def get_model_dense(x_train, y_train):
    print("get_model")

    x_input = tf.keras.Input(shape=x_train.shape[1:])
    x = x_input

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=800)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=y_train.shape[1])(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def conv_block(x, filters, kernel_size, activation):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size)(x)
    x = tf.keras.layers.Lambda(activation)(x)
    return x


def dense_block(x, units, activation, use_dropout=False):
    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.Lambda(activation)(x)

    if use_dropout:
        x = tf.keras.layers.Dropout(0.5)(x)

    return x


def get_model_ludo(x_train, y_train, input_kernel_size):
    print("get_model")

    x_input = tf.keras.Input(shape=x_train.shape[1:])
    x = x_input

    kernel_sizes = [(7, 7), (3, 3)]  # Define kernel sizes

    start_power = 5
    num_elements = len(kernel_sizes)  # Change this to the desired number
    filters = [pow(2, i) for i in range(start_power, start_power + num_elements)]

    # Convolutional layers
    for i, kernel_size in enumerate(kernel_sizes):
        x = conv_block(x, filters=filters[i], kernel_size=kernel_size, activation=tf.keras.activations.relu)

    # Flatten
    x = tf.keras.layers.Flatten()(x)

    # Dense layers
    x = dense_block(x, units=800, activation=tf.keras.activations.relu)

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
            x = tf.keras.layers.GroupNormalization(groups=8)(x)
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       dilation_rate=1,
                                       kernel_initializer=tf.keras.initializers.he_uniform)(x)
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.GroupNormalization(groups=8)(x_res)
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       dilation_rate=1,
                                       kernel_initializer=tf.keras.initializers.he_uniform)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x = tf.keras.layers.GroupNormalization(groups=8)(x)
        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   dilation_rate=1,
                                   kernel_initializer=tf.keras.initializers.he_uniform)(x)
        x = tf.keras.layers.Lambda(einops.rearrange,
                                   arguments={"pattern": "b (h h1) (w w1) c -> b h w (c h1 w1)", "h1": 2, "w1": 2})(x)

    for j in range(2):
        x = tf.keras.layers.GroupNormalization(groups=8)(x)
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   dilation_rate=1,
                                   kernel_initializer=tf.keras.initializers.he_uniform)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.GroupNormalization(groups=8)(x)
    x = tf.keras.layers.Dense(units=128,
                              kernel_initializer=tf.keras.initializers.he_uniform)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.GroupNormalization(groups=8)(x)
    x = tf.keras.layers.Dense(units=y_train.shape[1],
                              kernel_initializer=tf.keras.initializers.zeros)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def get_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    loss = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
    loss = tf.math.reduce_mean(loss)

    return loss


def get_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float64)
    y_pred = tf.cast(y_pred, dtype=tf.float64)

    accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    accuracy = tf.math.reduce_mean(accuracy)

    return accuracy


def train(model, optimiser, x_train, y_train):
    print("train")

    batch_size = 32

    for i in range(5):
        indices = list(range(x_train.shape[0]))
        np.random.shuffle(indices)

        for j in range(int(np.floor(x_train.shape[0] / batch_size))):
            current_index = batch_size * j

            current_x_train = []
            y_true = []

            for k in range(batch_size):
                current_x_train.append(x_train[current_index])
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

            print("Epoch: {0}, Iteration: {1}, Loss: {2}, Accuracy: {3}".format(str(i), str(j), str(loss.numpy()),
                                                                                str(accuracy.numpy())))

    return model


def train_gradient_accumulation(model, optimiser, x_train, y_train):
    print("train")

    batch_size = 32
    accumulated_divisor = 1

    accumulated_batch_size = int(np.floor(batch_size / accumulated_divisor))

    for i in range(5):
        indices = list(range(x_train.shape[0]))
        np.random.shuffle(indices)

        for j in range(int(np.floor(x_train.shape[0] / batch_size))):
            accumulated_gradients = [tf.zeros_like(trainable_variable) for trainable_variable in
                                     model.trainable_variables]

            current_index = batch_size * j

            losses = []
            accuracies = []

            for k in range(accumulated_batch_size):
                current_x_train = []
                y_true = []

                for m in range(batch_size):
                    current_x_train.append(x_train[current_index])
                    y_true.append(y_train[current_index])

                    current_index = current_index + 1

                current_x_train = tf.convert_to_tensor(current_x_train)
                y_true = tf.convert_to_tensor(y_true)

                with tf.GradientTape() as tape:
                    y_pred = model([current_x_train], training=True)

                    loss = get_loss(y_true, y_pred)

                gradients = tape.gradient(loss, model.trainable_weights)

                accumulated_gradients = [(accumulated_gradient + gradient) for accumulated_gradient, gradient in
                                         zip(accumulated_gradients, gradients)]

                losses.append(loss)
                accuracies.append(get_accuracy(y_true, y_pred))

            gradients = [gradient / accumulated_divisor for gradient in accumulated_gradients]
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            loss = tf.math.reduce_mean(losses)
            accuracy = tf.math.reduce_mean(accuracies)

            print("Epoch: {0}, Iteration: {1}, Loss: {2}, Accuracy: {3}".format(str(i), str(j), str(loss.numpy()),
                                                                                str(accuracy.numpy())))

    return model


def test(model, x_test, y_test):
    print("test")

    y_pred = model([x_test], training=True)

    loss = get_loss(y_test, y_pred)

    accuracy = get_accuracy(y_test, y_pred)

    print("Loss: {0}, Accuracy: {1}".format(str(loss.numpy()), str(accuracy.numpy())))

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

    # use dense
    model = get_model_ludo(x_train, y_train, input_kernel_size=7)
    model.summary()

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-04,
                                         weight_decay=1e-04)

    # train standard
    model = train(model, optimiser, x_train, y_train)

    _, y_pred = test(model, x_test, y_test)

    plot_test(x_test, y_test, y_pred)

    return True


if __name__ == "__main__":
    main()
