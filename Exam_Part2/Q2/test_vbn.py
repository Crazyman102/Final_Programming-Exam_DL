from matplotlib import image
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import (ResNet50,ResNet101, preprocess_input)
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score
import os
from tensorflow.keras.models import Model
from tensorflow import keras


batch_size = 32
img_size = 256
img_shape = (img_size, img_size)
test_path = 'test2/'


x_data_list = []
y_data_list = []
for roots, dirs, files in os.walk(test_path):
    for each in files:
        if each.find('checkpoint') == -1:
            x_data_list.append(os.path.join(roots.split("/")[-1], each))
            y_data_list.append(roots.split("/")[-1])


data_list = pd.DataFrame({})
data_list['img_path'] = x_data_list
data_list['label'] = y_data_list

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(dataframe=data_list,
                                                    directory=test_path,
                                                    x_col="img_path",
                                                    target_size=img_shape,
                                                    batch_size=batch_size,
                                                    class_mode=None,
                                                    shuffle=False)

model = keras.models.load_model('vbn.h5')

# y_pred_all = model.predict_generator(test_generator)
y_pred_all = model.predict(test_generator)
y_pred_all = y_pred_all.argmax(-1)
labels = {0 : 'type1', 1 : 'type2' ,2 : 'type3',3 : 'type4'}
predictions = [labels[k] for k in y_pred_all]

# export csv
df = pd.DataFrame()
df['img_path'] = x_data_list
df['predict'] = predictions
df.to_csv('test2.csv',header=None, index=False)
print(predictions)
