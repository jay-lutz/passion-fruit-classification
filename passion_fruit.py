from data import PassionFruitData
from alex_net import AlexNet
from mobile import MobileNetModified
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

architecture = 'alex'

if architecture == 'alex':
    alex = AlexNet()
    model = alex.alex_net_model()
elif architecture == 'resnet':
    model = ResNet50(include_top=True, weights=None, classes=3, classifier_activation='softmax')
elif architecture == 'mobile':
    mobile_modified = MobileNetModified()
    model = mobile_modified.get_model()
else:
    raise Exception('Invalid architecture.')

d = PassionFruitData(test_size=0.2, model=architecture)
train = d.train_data()
test = d.test_data()

if architecture == 'resnet':
    train['x'] = preprocess_input(train['x'])
    test['x'] = preprocess_input(test['x'])


model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001), loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(train['x'], train['y'], batch_size=16, epochs=25, validation_data=(test['x'], test['y']))
