import tensorflow as tf
from export_model import ExportModel

class MnistModel(tf.keras.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
    
    def call(self, inputs, training=False):
        x = self.reshape_(inputs)
        x = self.conv2d_relu1_(x)
        x = self.conv2d_relu2_(x)
        x = self.max_pooling_(x)
        if training:
            x = self.dropout1_(x)
        x = self.flatten_(x)
        x = self.dense1_(x)
        if training:
            x = self.dropout2_(x)
        x = self.dense2_(x)
        return x

    def build(self, input_shape=None):
        self.reshape_ = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv2d_relu1_ = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv2d_relu2_ = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu)
        self.max_pooling_ = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1_ = tf.keras.layers.Dropout(0.25)
        self.flatten_ = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1_ = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dropout2_ = tf.keras.layers.Dropout(0.5)
        self.dense2_ = tf.keras.layers.Dense(10, activation='softmax')

    @tf.function(input_signature=[
        tf.TensorSpec([None,None,None,3], dtype='float32', name='input_image'),
    ])
    def serving_fn(self, input_images):
        return {
            'prediction' : self.call(input_images, training=False)
        }

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = MnistModel()
    model.build(input_shape=None)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)
    tf.keras.models.save_model(model, "./mnist/saved_model")