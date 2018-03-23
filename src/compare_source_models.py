from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Conv2D, MaxPooling2D, Input
from keras import metrics
import time
import cv2


img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
epochs = 30
batch_size = 20
output_classes = 2
n200_data_dir = 'data/n200'
n50_data_dir = 'data/n50'
n25_data_dir = 'data/n25'
validation_data_dir = 'data/test'

val_samples = 100
val_steps = val_samples / batch_size
n = 200
train_samples = 2*n
train_steps = train_samples / batch_size


def instantiate_model(source_model):
    inp = Input(shape=(img_width, img_height, 3), name='input_image')
    main_model = source_model
    for layer in main_model.layers:
        layer.trainable=False

    main_model = main_model(inp)
    main_out = Flatten()(main_model)
    main_out = Dense(256, activation='relu')(main_out)
    main_out = Dropout(.3)(main_out)
    main_out = Dense(output_classes, activation='softmax')(main_out)

    model = Model(input=inp, output=main_out)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

print('\n==== Source model: VGG-16')
t0 = time.time()
vgg_source = applications.VGG16(weights='imagenet', include_top=False)
t1 = time.time()
vgg_complete = instantiate_model(vgg_source)
print(vgg_source.summary())
print(vgg_complete.summary())
print('Source model load time: {}s'.format(t1-t0))

print('\n==== Source model: InceptionV3')
t0 = time.time()
inception_source = applications.InceptionV3(weights='imagenet', include_top=False)
t1 = time.time()
inception_complete = instantiate_model(inception_source)
print(inception_source.summary())
print(inception_complete.summary())
print('Source model load time: {}s'.format(t1-t0))

print('\n==== Source model: ResNet50')
t0 = time.time()
resnet_source = applications.ResNet50(weights='imagenet', include_top=False)
t1 = time.time()
resnet_complete = instantiate_model(resnet_source)
print(resnet_source.summary())
print(resnet_complete.summary())
print('Source model load time: {}s'.format(t1-t0))

print('\n==== Source model: MobileNet')
t0 = time.time()
mobilenet_source = applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
t1 = time.time()
mobilenet_complete = instantiate_model(mobilenet_source)
print(mobilenet_source.summary())
print(mobilenet_complete.summary())
print('Source model load time: {}s'.format(t1-t0))

print('\n==== Source model: NASNet (Mobile)')
t0 = time.time()
nasnet_source = applications.NASNetMobile(weights='imagenet', include_top=False)
t1 = time.time()
nasnet_complete = instantiate_model(nasnet_source)
print(nasnet_source.summary())
print(nasnet_complete.summary())
print('Source model load time: {}s'.format(t1-t0))

print('\n==== Loading data')
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20.,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

n200_generator = train_datagen.flow_from_directory(
    n200_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
print('==== Loaded data\n')


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


print('\n==== Fitting models')
print('\n==== Source model: VGG-16')
t0 = time.time()
vgg_hist = vgg_complete.fit_generator(
    n200_generator,
    steps_per_epoch=train_steps,#nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps)
    #callbacks=[WandbKerasCallback()]
t1 = time.time()
vgg_complete.save('results/sources/vgg.h5')
print('==== time taken to fit model: {}s'.format(t1 - t0))

print('\n==== Source model: InceptionV3')
t0 = time.time()
inception_hist = inception_complete.fit_generator(
    n200_generator,
    steps_per_epoch=train_steps,#nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps)
    #callbacks=[WandbKerasCallback()]
t1 = time.time()
inception_complete.save('results/sources/inception.h5')
print('==== time taken to fit model: {}s'.format(t1 - t0))

print('\n==== Source model: ResNet50')
t0 = time.time()
resnet_hist = resnet_complete.fit_generator(
    n200_generator,
    steps_per_epoch=train_steps,#nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps)
    #callbacks=[WandbKerasCallback()]
t1 = time.time()
resnet_complete.save('results/sources/resnet.h5')
print('==== time taken to fit model: {}s'.format(t1 - t0))
print('==== models fit\n')

print('\n==== Source model: MobileNet')
t0 = time.time()
mobilenet_hist = mobilenet_complete.fit_generator(
    n200_generator,
    steps_per_epoch=train_steps,#nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps)
    #callbacks=[WandbKerasCallback()]
t1 = time.time()
mobilenet_complete.save('results/sources/mobilenet.h5')
print('==== time taken to fit model: {}s'.format(t1 - t0))
print('==== models fit\n')

print('\n==== Source model: NASNet (Mobile)')
t0 = time.time()
nasnet_hist = nasnet_complete.fit_generator(
    n200_generator,
    steps_per_epoch=train_steps,#nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps)
    #callbacks=[WandbKerasCallback()]
t1 = time.time()
nasnet_complete.save('results/sources/nasnet.h5')
print('==== time taken to fit model: {}s'.format(t1 - t0))
print('==== models fit\n')

print('\n==== outputting histories')
import pickle

hists = [vgg_hist, inception_hist, resnet_hist, mobilenet_hist, nasnet_hist]
names = ['vgg_hist.p', 'inception_hist.p', 'resnet_hist.p', 'mobilenet_hist.p', 'nasnet_hist.p']

for i in zip(names, hists):
    name = i[0]
    hist = i[1].history
    path = 'results/sources/{}'.format(name)
    print(type(hist))
    with open(path, 'wb') as output:
        pickle.dump(hist, output, -1)
    print('outputted {} to {}'.format(name.split('.')[0], path))
print('==== histories outputted\n')
