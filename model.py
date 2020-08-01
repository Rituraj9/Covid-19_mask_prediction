import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_df = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

#Preprocessing the Test Set
test_datagen = ImageDataGenerator(rescale=1./255)
test_df = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

cnn.fit(x=train_df,validation_data=test_df,epochs=25)

cnn.save('covid_model.pkl')
mm = tf.keras.models.load_model('covid_model.pkl')

#making Single prediction
from keras.preprocessing import image
import numpy as np
test_image = image.load_img('dataset/single_prediction/2.jpg',target_size=(64,64))
#convert format to array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = mm.predict(test_image)
print(result)

print(train_df.class_indices)
if result[0][0] == 1:
    predictions = 'Without mask'
else:
    predictions = 'With Mask'
    
print(predictions)