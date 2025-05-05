#预测
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
#测试图片地址
path = 'dataset/predicted/dog1.jpeg'
#加载模型
model = load_model('model.h5')
#将图片转换成 150*150 的格式，与模型训练的输入保持一致


img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img) / 255.0
#在第 0 维添加维度变为 1x150x150x3，和我们模型的输入数据一样
x = np.expand_dims(x, axis=0)
#np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组，我们一次只有一个数据所以不这样也可以
images = np.vstack([x])

batch_size #批量大小，程序会分批次地预测测试数据，这样比每次预测一个样本会快
classes = model.predict(images, batch_size=1)
classes[0]#表示分类概率，大于 0.5 表示分类为 dog，小于 0.5 表示分类为 cat

print(classes[0])
if classes[0] > 0.5:
    print("It is a dog")
else:
    print("It is a cat")

