from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import layers,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d','--dataset',type=str, required=True,help='path to the dataset')
ap.add_argument('-p','--plot',type=str, default='plot.png',help='path to the output plot')
ap.add_argument('-m','--model',type=str,default='mask_detector.model',help='path to the output model')
args= ap.parse_args()

INIT_LR=1e-4
EPOCHS=10
BS=32

print('loading images:')
print(args)
imagePaths= list(paths.list_images(args.dataset))
data= []
labels= []

for imagePath in imagePaths:
    label= imagePath.split(os.path.sep)[-2]
    image= load_img(imagePath, target_size=(224,224))
    image= img_to_array(image)
    image= preprocess_input(image)

    data.append(image)
    labels.append(label)


lb= LabelBinarizer()
labels= lb.fit_transform(labels)
labels= to_categorical(labels)

data= np.array(data, dtype='float32')
labels= np.array(labels)

(train_X, test_X, train_Y, test_Y)= train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=19)
print("train data size:",len(train_X),"test data size", len(test_X))
aug= ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

baseModel= MobileNetV2(weights='imagenet',include_top=False, input_tensor=layers.Input(shape=(224,224,3)))

headModel= baseModel.output
headModel= layers.AveragePooling2D(pool_size=(7,7))(headModel)
headModel= layers.Flatten(name='flatten')(headModel)
headModel= layers.Dense(128, activation='relu')(headModel)
headModel= layers.Dropout(0.5)(headModel)
headModel= layers.Dense(2,activation='softmax')(headModel)

model= Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
	layer.trainable= False

print("model complie:")
optimizer = Adam(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)
callback=EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.005,
    patience=1,
    restore_best_weights=True
)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(
	aug.flow(train_X, train_Y, batch_size=BS),
	validation_data=(test_X, test_Y),
	epochs=EPOCHS,
    callbacks=[callback])

print("model evaluation:")
predIdxs = model.predict(test_X, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(test_Y.argmax(axis=1), predIdxs, target_names=lb.classes_))

print("save model to disk")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
epoch = len(history.history["val_loss"])
plt.style.use("ggplot")
plt.figure()
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.plot(np.arange(0, epoch), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epoch), history.history["val_accuracy"], label="val_acc")
plt.legend(loc="lower left")
plt.savefig("plot.png")