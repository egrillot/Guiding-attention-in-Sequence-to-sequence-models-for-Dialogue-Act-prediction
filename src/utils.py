from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Accuracy
from keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
   if epoch < 20:
     return lr
   else:
     return lr * 0.5

def compile_fit(model):
  model.compile(optimizer=Adam(learning_rate=0.01, clipnorm=5.), loss=categorical_crossentropy(), metrics=Accuracy())
  callbacks = LearningRateScheduler(scheduler)
  model.fit(x=[callers_train, conv_train], y=tags_train, epochs=500, validation_split=.2, callbacks=[callbacks], shuffle=True)

def test(model):
  model.test(x=[callers_test, conv_test], y=tags_test)

def benchmark():
  #find paper results

  models = {'VGRUe-VGRUd': VGRUd(VGRUe()),
            'HGRU-VGRUd': VGRUd(HGRU()),
            'PersoHGRU-VGRUd': VGRUd(PersoHGRU()),
            'VGRUe-VGRUatt': VGRUatt(VGRUe()),
            'HGRU-VGRUatt': VGRUatt(HGRU()),
            'PersoHGRU-VGRUatt': VGRUatt(PersoHGRU()),
            'VGRUe-VGRUhga': VGRUhga(VGRUe()),
            'HGRU-VGRUhga': VGRUhga(HGRU()),
            'PersoHGRU-VGRUhga': VGRUhga(PersoHGRU()),
            'VGRUe-VGRUsga': VGRUsga(VGRUe()),
            'HGRU-VGRUsga': VGRUsga(HGRU()),
            'PersoHGRU-VGRUsga': VGRUsga(PersoHGRU()),
              }

  for model in models:
    print('Training {} model ...'.format(model))
    compile_fit(models[model])
    print('Testing {} model ...'.format(model))
    test(models[model])