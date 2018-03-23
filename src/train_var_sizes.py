from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input, BatchNormalization
from keras import metrics
import time
import cv2

img_width, img_height = 224, 224

batch_size = 10
output_classes = 2
n200_data_dir = '../data/n200'
n150_data_dir = '../data/n150'
n100_data_dir = '../data/n100'
n50_data_dir = '../data/n50'
n25_data_dir = '../data/n25'
validation_data_dir = '../data/test'
val_samples = 100
val_steps = val_samples / batch_size


def instantiate_model(main_model):
    inp = Input(shape=(img_width, img_height, 3), name='input_image')

    for layer in main_model.layers:
        layer.trainable=False

    main_model = main_model(inp)
    main_out = Dense(output_classes, activation='softmax')(main_model)

    model = Model(input=inp, output=main_out)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


print('\n==== Instantiating source models')
vgg_source = applications.VGG16(weights='imagenet',input_shape=(224,224,3), pooling='max', include_top=False)
inception_source = applications.InceptionV3(weights='imagenet', input_shape=(224,224,3), pooling='max', include_top=False)
mobilenet_source = applications.MobileNet(weights='imagenet', include_top=False,pooling='max', input_shape=(224, 224, 3))
print('==== Instantiated source models\n')

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

n150_generator = train_datagen.flow_from_directory(
    n150_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

n100_generator = train_datagen.flow_from_directory(
    n100_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

n50_generator = train_datagen.flow_from_directory(
    n50_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

n25_generator = train_datagen.flow_from_directory(
    n25_data_dir,
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
source_models = [mobilenet_source, vgg_source, inception_source]
model_names = ['MobileNet', 'VGG16', 'InceptionV3']
generators = [n25_generator, n50_generator, n100_generator, n150_generator, n200_generator]
ns = [25, 50, 100, 150, 200]
epochs = [20,40]
hist_list = []

for (src, src_name) in zip(source_models, model_names):
    model = instantiate_model(src)
    for (n, train_generator) in zip(ns, generators):
        train_steps = 2 * n / batch_size
        for e in epochs:
            print('==== Source: {}\tImages per class: {}\tEpochs: {}'.format(src_name, n, e))
            t0 = time.time()
            hist = model.fit_generator(
                train_generator,
                steps_per_epoch=train_steps,#nb_train_samples,
                epochs=e,
                validation_data=validation_generator,
                validation_steps=val_steps)
            t1 = time.time()
            t = t1 - t0
            hist_dict = {
                'source_model':src_name,
                'images_per_class':n,
                'epochs':e,
                'hist':hist.history,
                'training_time':t
                }
            hist_list.append(hist_dict)
            path = '/home/ubuntu/image_classification/results/circ/{}_n{}_e{}.h5'.format(src_name, n, e)
            model.save(path)
print('==== models fit\n')


print('\n==== outputting histories')
import pickle
path = '/home/ubuntu/image_classification/results/circ/circ.p'
with open(path, 'wb') as output:
    pickle.dump(hist, output)
print('\n==== outputted histories to {}'.format(path))
