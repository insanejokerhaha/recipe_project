import numpy as np
import tensorflow as tf
import random
import os
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from datetime import datetime

#data_directory = 'binary_classification_norep'
data_directory = 'data/Ecare_embedding/docrep_improve'
x_train = np.load(f'{data_directory}/X_train.npy')
x_test = np.load(f'{data_directory}/X_test.npy')
y_train = np.load(f'{data_directory}/y_train.npy')
y_test = np.load(f'{data_directory}/y_test.npy')

print(len(x_train),len(x_test), (len(x_train)+len(x_test)))
print(x_train.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1600, input_shape=(1600,), activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

"""model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(500, input_shape=(500,), activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])"""

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## Callbacks:
# Create the TensorBoard callback


batch_size = 256
epochs = 100

class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('loss')<0.4):

      print("\nReached 60% accuracy so cancelling training!")

      self.model.stop_training = True

#callbacks = myCallback()

# Set up logging directory
## Use date-time as logdir name:
#dt = datetime.now().strftime("%Y%m%dT%H%M")
#logdir = os.path.join("logs",dt)
logdir = 'logs/docrep_improve'
if not os.path.exists(logdir):
    os.mkdir(logdir)


## Callbacks:
# Create the TensorBoard callback
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1,
    batch_size=batch_size,
    write_graph=True,
    write_grads=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=0
)

# Training logger
csv_log = os.path.join(logdir, 'training.csv')
csv_logger = CSVLogger(csv_log, separator=',', append=True)

# Only save the best model weights based on the val_loss
checkpoint = ModelCheckpoint(os.path.join(logdir, 'simple_model-{epoch:02d}-{val_loss:.2f}.h5'),
                             monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=True,
                             mode='auto')


# Save the embedding mode weights based on the main model's val loss
# This is needed to reecreate the emebedding model should we wish to visualise
# the latent space at the saved epoch
class SaveEmbeddingModelWeights(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.best = np.Inf
        self.filepath = filepath
        self.no_update_count = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("SaveEmbeddingModelWeights requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.best:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            # if self.verbose == 1:
            # print("Saving embedding model weights at %s" % filepath)
            model.save_weights(filepath, overwrite=True)
            self.best = current
        else:
            self.no_update_count += 1

        if self.no_update_count >= 3:
            print("\nConsecutive 3 epochs no decrease in val_loss, stop training")
            self.model.stop_training = True


# Save the embedding model weights if you save a new snn best model based on the model checkpoint above
emb_weight_saver = SaveEmbeddingModelWeights(os.path.join(logdir, 'emb_model-{epoch:02d}.h5'))

callbacks = [tensorboard, csv_logger, checkpoint, emb_weight_saver]

model.fit(x_train, y_train, epochs=epochs, callbacks=[callbacks], validation_split=0.2)#validation_data=(x_test,y_test)

model.evaluate(x_test,  y_test, verbose=2)

model_path = 'saved_model'
if not os.path.exists(model_path):
    os.mkdir(model_path)

model.save(f'{model_path}/docrep_improve')