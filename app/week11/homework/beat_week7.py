"""
    Try to beet your result on survival dataset from week 7 with a neural network.
    You can use only the layers, activations and losses that you are familiar from
    our course (you will pass some during this week, and you can use them afterwards).
    Send either google colab link or .py file including the tf model creation, training
    and testing codes. Do not use test data for model selection, do it only once on the
    best network you choose and compare with the score you have achieved during week 7
    team practice.
"""


import tensorflow as tf
from tensorflow import keras
from keras import layers


model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
