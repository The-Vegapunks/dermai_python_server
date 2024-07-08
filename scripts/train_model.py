# Keras API functions
from keras.api.applications.vgg19 import VGG19
from keras.api.utils import image_dataset_from_directory
from keras.api. layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, RandomBrightness, RandomFlip, RandomRotation
from keras.api.models import Sequential, save_model
from keras.api.optimizers import Adam
from keras.api.callbacks import  EarlyStopping, ModelCheckpoint
import warnings

warnings.filterwarnings('ignore')

img_height = 400
img_width = 400

train_dir = 'kaggle/train'
val_dir = 'kaggle/val'

train_data = image_dataset_from_directory(train_dir, label_mode='categorical', image_size=(img_width, img_height), batch_size=16, shuffle=True, seed=42)
val_data = image_dataset_from_directory(val_dir, label_mode='categorical', image_size=(img_width, img_height), batch_size=16, shuffle=True, seed=42)


augmentation = Sequential()
augmentation.add(RandomBrightness(factor=0.1))
augmentation.add(RandomFlip(mode='horizontal_and_vertical'))
augmentation.add(RandomRotation(factor = 0.2))

vgg19 = VGG19(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
# vgg19.summary()

vgg19.trainable = False

set_true = False

for layer in vgg19.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
        set_true = True
    if set_true:
            layer.trainable = True 
            
# for layer in vgg19.layers:
#     print(f'Layer Name = {layer.name},      Trainable = {layer.trainable}')


model = Sequential()
model.add(augmentation)
model.add(vgg19)
# model.add(Flatten())
model.add(Dropout(0.3))
# model.add(Dense(3000, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1500, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(500, activation = 'relu'))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))          
model.add(Dense(6, activation = 'softmax'))

adam = Adam(learning_rate = 0.0001)


model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

model.build((None, 400, 400, 3))
model.summary()

# modify the path to the desired name
chk_path = 'model/archive_model.keras'

checkpoint = ModelCheckpoint(filepath = chk_path, monitor='val_accuracy', mode='max', save_best_only = True, verbose = 1)
model.load_weights(chk_path)

early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 5, mode = 'max', verbose = 1)
history = model.fit(train_data, validation_data = val_data, epochs = 100, batch_size = 8, callbacks = [early_stopping, checkpoint])

# modify the path to the desired name
save_model(model=model, filepath=r"model/new_save.keras")