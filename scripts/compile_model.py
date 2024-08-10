from keras.api.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, RandomBrightness, RandomFlip, RandomRotation
from keras.api.models import Sequential
from keras.api.applications.vgg19 import VGG19
from keras.api.optimizers import Adam
from keras.api.utils import img_to_array
from PIL import Image
import numpy as np

class Model:
    def __init__(self):
        self.model = Sequential()
        self.init_model('model/bestmodel.keras')

    def init_model(self, weight_path):
        img_width = img_height = 400

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


        self.model.add(augmentation)
        self.model.add(vgg19)
        # self.model.add(Flatten())
        self.model.add(Dropout(0.3))
        # self.model.add(Dense(3000, activation = 'relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(1500, activation = 'relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(500, activation = 'relu'))
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(100, activation = 'relu'))          
        self.model.add(Dense(6, activation = 'softmax'))

        adam = Adam(learning_rate = 0.0001)


        self.model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

        self.model.build((None, 400, 400, 3))

        self.model.load_weights(weight_path)

    def predict_class(self, img_data):
        """
        0 - Enfeksiyonel
        1 - Ekzama
        2 - Akne
        3 - Pigment
        4 - Benign
        5 - Malign
        """
        img_width, img_height = 400,400

        predict_img = Image.open(img_data).convert('RGB')
        
        # predict_img = load_img(img)
        predict_img = predict_img.resize((img_width, img_height))
        predict_img = img_to_array(predict_img)
        
        flatten_predict_img = predict_img.reshape(-1, 400,400,3)
        
        predicted = self.model.predict(flatten_predict_img, batch_size=1)
        # array of probability it is the dieases
        converted_values = [float(value) for value in predicted[0]]
        predicted_name = np.argmax(converted_values)
        
        class_name = [(1, 'Enfeksiyonel'), (2, 'Ekzama'), (3, 'Akne'), (4, 'Pigment'), (5, 'Benign'), (6, 'Malign')]
        
        for index in range(0, len(class_name)):
            print(f"{class_name[index]} --> {converted_values[index]}")


        return class_name[predicted_name]
 
