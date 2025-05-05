import os
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#数据集地址，如果出现地址错误，则考虑使用绝对地址
base_dir = 'dataset/cats_and_dogs'
#指定每一种数据的位置
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

#Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

#Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#模型处添加了 dropout 随机失效，也就是说有时候可能不用到某些神经元，失效率为0.5

#下面是模型的整体结构，可以观察到每一层卷积之后，都会使用一个最大池化层对提取的数据进行降维，减少计算量，后续实验修改网络结构主要修改下面部分
# 设计模型
model = tf.keras.models.Sequential([
#我们的数据是 150x150 而且是三通道的，所以我们的输入应该设置为这样的格式。
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2), # 最大池化层
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5), # dropout 层通通过忽略一般数量特征，可以减少过拟合现象
    tf.keras.layers.Flatten(), # 全链接层，将多维输入一维化
    tf.keras.layers.Dense(512, activation='relu'),
#二分类只需要一个输出
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#
#进行优化方法选择和一些超参数设置因为只有两个分类。所以用 2 分类的交叉熵，使用 RMSprop，学习率为 0.0001.优化指标为 accuracy
model.compile(loss='binary_crossentropy',# 损失函数使用交叉熵
    optimizer=RMSprop(lr=1e-4), # 优化器，学习率设置为 0.0001
    metrics=['acc'])

#数据处理把每个数据都放缩到 0 到 1 范围内
#这里的代码进行了更新，原来这里只进行归一化处理，现在要进行数据增强。
train_datagen = ImageDataGenerator(
rescale=1. / 255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)

#生成训练集的带标签的数据
train_generator = train_datagen.flow_from_directory(train_dir, # 训练图片的位置
batch_size=20, # 每一个投入多少张图片训练
class_mode='binary', # 设置我们需要的标签类型
target_size=(150, 150)) # 将图片统一大小

#生成验证集带标签的数据
validation_generator = test_datagen.flow_from_directory(validation_dir, # 验证图片的位置
batch_size=20, # 每一个投入多少张图片训练
class_mode='binary', # 设置我们需要的标签类型
target_size=(150, 150)) # 将图片统一大小

#进行训练
history = model.fit_generator(
train_generator, # 训练集数据
steps_per_epoch=20, # 每个 epoch 训练多少次
epochs=10, # 训练轮数，建议在[10,50]如果电脑训练速度快，可以大于 50
validation_data=validation_generator, # 验证集数据
validation_steps=10,
verbose=1) # 训练进度显示方式，可取值 0，1（显示训练进度条），2（一个 epoch输出一条信息）
#保存训练的模型到当前目录
model.save('model.h5')
#得到精度和损失值
acc = history.history['acc'] # train_acc
val_acc = history.history['val_acc'] # val_acc
loss = history.history['loss'] # train_loss
val_loss = history.history['val_loss'] # val_loss
epochs = range(len(acc)) # 得到迭代次数

#绘制精度曲线
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.legend(('Training accuracy', 'validation accuracy'))
plt.figure()

#绘制损失曲线
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.legend(('Training loss', 'validation loss'))
plt.title('Training and validation loss')
plt.show()