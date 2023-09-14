import os
import numpy as np

try:
    os.add_dll_directory("C:\\Programme\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
    print("Cuda importing finished")
except:
    print("Cuda importing failed")


import tensorflow as tf
from tensorflow import keras as kr
from tensorflow import math
from tensorflow.keras import layers
from tensorflow.data import Dataset
from matplotlib import pyplot as plt


def create_dataset(foldername):
    type = np.load(os.path.join(foldername,"Type.npy"))

    dataset_length = type.shape[0]
    dataset_digits = len(str(dataset_length))

    for i in range(dataset_length):
        picname = "LEEDImage_%s%s.jpg" %((dataset_digits-len(str(i)))*"0",i)
        rgb = np.array(plt.imread(os.path.join(foldername,picname)),dtype=np.float64)
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        picture = 0.2989 * r + 0.5870 * g + 0.1140 * b
        yield np.expand_dims(picture,2), np.expand_dims(type[i,:],1)



picSpec = tf.TensorSpec(shape=(1000,1000,1),dtype=tf.float64)
typeSpec = tf.TensorSpec(shape=(4,1),dtype=tf.float64)

if __name__=="__main__":
    
    dataset = Dataset.from_generator(lambda: create_dataset("Training3"),
                                    output_signature=(picSpec,typeSpec))
    dataset_batched = dataset.batch(32)
    val_dataset = Dataset.from_generator(lambda: create_dataset("Validation3"),
                                    output_signature=(picSpec,typeSpec))
    val_dataset_batched = dataset.batch(32)


    input = layers.Input(shape=(1000,1000,1))
    pool0 = layers.AveragePooling2D((2,2))(input)
    conv1 = layers.Conv2D(filters=32,kernel_size=(5,5),padding="same",activation="relu")(pool0)
    pool1 = layers.AveragePooling2D((2,2))(conv1)
    conv2 = layers.Conv2D(filters=64,kernel_size=(5,5),padding="same",activation="relu")(pool1)
    pool2 = layers.MaxPool2D((5,5))(conv2)
    conv3 = layers.Conv2D(filters=128,kernel_size=(5,5),padding="same",activation="relu")(pool2)
    pool3 = layers.MaxPool2D((2,2))(conv3)
    flatten = layers.Flatten()(pool3)

    pre = layers.Dense(20,activation="relu")(flatten)
    type = layers.Dense(4,activation="softmax")(pre)


    LEEDmodel = kr.Model(inputs=input,outputs=type)
    LEEDmodel.compile(optimizer=kr.optimizers.Adam(learning_rate=kr.optimizers.schedules.ExponentialDecay(0.001,200,0.5)),
                    loss=[kr.losses.CategoricalCrossentropy()],
                    metrics=kr.metrics.MeanAbsoluteError(name="mse"))
    LEEDmodel.summary()
    tensorboard_callback = kr.callbacks.TensorBoard(histogram_freq=1,write_graph=False,write_images=True)

    LEEDmodel.fit(dataset_batched,validation_data=val_dataset_batched,epochs=8,callbacks=tensorboard_callback)
    LEEDmodel.save("LEEDmodel")





