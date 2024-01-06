import numpy
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from segmentation_models import Unet
from segmentation_models.metrics import IOUScore
from segmentation_models.losses import JaccardLoss
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = './data/'

img_size = (128,128)
batch_size = 64
seed = 1

image_datagen = ImageDataGenerator(rescale=1./255)
mask_datagen = ImageDataGenerator(rescale=1./255)

#train images and their masks
train_image_generator = image_datagen.flow_from_directory(
                        path,
                        classes=['xtrain'],
                        target_size=img_size,
                        batch_size=batch_size,
                        class_mode=None, # None because we dont want labels to be returned, just images.
                        seed=seed, 
                        shuffle=False,
                        color_mode='rgb')

train_mask_generator = mask_datagen.flow_from_directory(
                        path,
                        classes=['ytrain'],
                        target_size=img_size,
                        batch_size=batch_size,
                        class_mode=None,# None because we dont want labels to be returned, just images.
                        seed=seed,
                        shuffle=False,
                        color_mode='grayscale')

train_generator = zip(train_image_generator, train_mask_generator)

# Validation images and their masks
# created a separate validation data of 1000 images and masks taken from the training data.
# xval and yval folders contain those images and masks respectively.
val_image_generator = image_datagen.flow_from_directory(
                        path,
                        classes=['xval'],
                        target_size=img_size,
                        batch_size=batch_size,
                        class_mode=None,# None because we dont want labels to be returned, just images.
                        seed=seed,
                        shuffle=False,
                        color_mode='rgb')

val_mask_generator = mask_datagen.flow_from_directory(
                        path,
                        classes=['yval'],
                        target_size=img_size,
                        batch_size=batch_size,
                        class_mode=None, # None because we dont want labels to be returned, just images.
                        seed=seed,
                        shuffle=False,
                        color_mode='grayscale')

validation_generator = zip(val_image_generator, val_mask_generator)

# Test images and their masks
test_image_generator = image_datagen.flow_from_directory(
                        path,
                        classes=['xtest'],
                        target_size=img_size,
                        batch_size=batch_size,
                        class_mode=None, # None because we dont want labels to be returned, just images.
                        seed=seed,
                        shuffle=False,
                        color_mode='rgb')

test_mask_generator = mask_datagen.flow_from_directory(
                        path,
                        classes=['ytest'],
                        target_size=img_size,
                        batch_size=batch_size,
                        class_mode=None, # None because we dont want labels to be returned, just images.
                        seed=seed,
                        shuffle=False,
                        color_mode='grayscale')

test_generator = zip(test_image_generator, test_mask_generator)

print("train_image_gen. shape: ",len(train_image_generator)) # 17697/64 = 277 (batchsize)
print("train_mask_gen. shape: ",len(train_mask_generator))
print("train_image_gen.[0] shape: ",train_image_generator[0].shape) #each batch shape
print("train_mask_gen.[0] shape: ",train_mask_generator[0].shape)

print("\nval_image_gen. shape: ",len(val_image_generator)) #1001/64 = 16
print("val_image_gen.[0] shape: ",val_image_generator[0].shape) #each batch shape
print("val_mask_gen.[0] shape: ",val_mask_generator[0].shape)

print("\ntest_image_gen. shape: ",len(test_image_generator)) #3740/64 = 58.4
print("test_image_gen.[0] shape: ",test_image_generator[0].shape) #each batch shape
print("test_mask_gen.[0] shape: ",test_mask_generator[0].shape)

# Model 
# Since our principle class is "Person", and encoder- backbone ie VGG has been pre-trained for this class
# it is better to freeze the weights.

model = Unet(encoder_freeze=True)
model.summary()

# Loss function and metric taken from the segmentation-models library itself.
loss_func = JaccardLoss(per_image=True)
metric = IOUScore(per_image=True)

# Early stopping callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.compile(loss = loss_func, optimizer="adam", metrics=[metric])

model_history = model.fit(train_generator, steps_per_epoch = 277, validation_data=validation_generator,
           validation_steps = 16, batch_size=64, epochs=5, callbacks=[callback])

model.evaluate(test_generator, batch_size=64, steps=59, verbose=1) #len(test_gen is 59)

model.save('Unet_model')


path_test_img = './data/xtest/0014.png'

img_test_orig = cv2.imread(path_test_img)
h, w = img_test_orig.shape[0:2]

img_test_resized = cv2.resize(img_test_orig, (128,128))
img_test = np.asarray(img_test_resized)/255.0
img_test = img_test[np.newaxis,...]
print(img_test.shape)

# Predicting the output from the trained model.
pred_img = model.predict(img_test)
print(pred_img.shape)

pred_img = np.squeeze(pred_img) #removing extra 1-dimensions.

result = img_test_resized.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)

pred_img_copy = pred_img.copy()
pred_img_copy[pred_img_copy<0.5] = 0
pred_img_copy[pred_img_copy>=0.5] = 255 #binarising
print("Shape of result: " , result.shape)
print("Shape of pred_img_copy: " , pred_img_copy.shape)

result[:, :, 3] = pred_img_copy # adding mask in alpha channel
result = cv2.resize(result, (h,w)) # resizing back to original size
print("bg_removed shape: " , result.shape)

fig = plt.figure(figsize=(20, 15))
  
# setting values to rows and column variables
rows = 5
columns = 5

fig.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(img_test_orig, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title("Original Image")

fig.add_subplot(rows, columns, 2)
plt.imshow(pred_img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title("Predicted Mask")

fig.add_subplot(rows, columns, 3)
plt.imshow(result)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title("Final Output")

plt.show()