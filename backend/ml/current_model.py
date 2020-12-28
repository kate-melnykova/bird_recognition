import tensorflow as tf

model_weights = '../data/Mobilenet-99.05.h5'


model = tf.keras.Sequential()
#baseline = tf.keras.applications.MobileNet(input_shape=(224, 224, 3),
#                                        include_top=False)
baseline = tf.saved_model.load(model_weights)
model.add(baseline)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(230, activation='softmax'))
model.load_weights(model_weights)