import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
print(x_train.shape)

# x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
# x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
# print(x_train.shape)

# x_train = tf.expand_dims(x_train, axis=3, name=None)
# x_test = tf.expand_dims(x_test, axis=3, name=None)
# print(x_train.shape)

# x_val = x_train[-10000:,:,:,:]
# y_val = y_train[-10000:]
# print(x_val.shape)
# x_train = x_train[:-10000,:,:,:]
# y_train = y_train[:-10000]
# print(x_train.shape)
num_samples = x_train.shape[0]
batch_size = 100

model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='tanh'))
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# with tf.device('/CPU:0'):
model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=5, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)