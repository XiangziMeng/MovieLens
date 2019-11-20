#coding: utf-8

import tensorflow as tf
from utils import *

if __name__ == "__main__":
    input_size, output_size = 3592, 6040
    data_generator = get_data_generator("data/ratings.dat", 1000)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(output_size, activation='softmax')])
  
    model.compile(optimizer='adam',
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])
     
    model.fit_generator(data_generator, steps_per_epoch=100, epochs=500)
    layer_0 =model.layers[0]
    layer_1 =model.layers[1]
    weights_0 = layer_0.get_weights()[0]
    weights_1 = layer_1.get_weights()[0].T

    fp_0 = open("movie_embeddings.txt", "w")
    for i in range(input_size):
        embedding = weights_0[i]
        fp_0.write("%d\t%s\n" % (i, ",".join(map(str, embedding))))
    fp_0.close()
    fp_1 = open("user_embeddings.txt", "w")
    for i in range(output_size):
        embedding = weights_1[i]
        fp_1.write("%d\t%s\n" % (i, ",".join(map(str, embedding))))
    fp_1.close()
