# 1. Exploring
# 2. Data preparation
#	 2.1 Load data
#	 2.2 Check for null and missing values
#	 2.3 Normalization
#	 2.4 Reshape
#	 2.5 Label encoding
#	 2.6 Split training and valdiation set
# 3. CNN
# 	3.1 Define the model
# 	3.2 Set the optimizer and annealer
# 	3.3 Data augmentation
# 4. Evaluate the model
# 	4.1 Training and validation curves
# 	4.2 Confusion matrix
# 5. Prediction and submition
# 	5.1 Predict and Submit results

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler

# from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from keras.optimizers import RMSprop
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ReduceLROnPlateau


####################		Loading data
print('\n\t\t\t START OFF \n')
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
# print(train.head(5))
# print(train.shape)		# (7049, 31)


####################		Divide data into 2 groups
image    = train['Image']
features = train.drop(labels = ['Image'], axis = 1)
# see = features.columns.tolist()
# for s in see:
# 	print(s)
# print('\n')
# print(features)
# print(type(image[0]))		# str
# print(image.shape)		# (7049,)
# print(features.shape)		# (7049, 30)
# print(image.shape[0])




#################### 	MISSING DATA

#		Check for null and missing values
print('\n\t Part 1')
missing_1 = image.isnull().sum()
# print(missing_1)
# print(image.isnull().any())

print('\n\t Part 2')
missing_2 = features.isnull().sum()
missing_2 = missing_2[missing_2 > 0]
missing_2.sort_values(inplace=True)
# missing_2.plot.bar()
# plt.tight_layout()
# plt.show()
# plt.gcf().clear()
# print(missing_2)
# print(features.isnull().any().describe())
# print(features.isnull().count())




#################### 	NORMALIZATION

# print(features.head(5))
features = np.log1p(features)
features.fillna(0, inplace=True)
# print('\n\t\tNormalization\n')
# print(features)




#################### 	SHAPE

# 		Я РАБОТА с КАРТИНКОЙ! ДЕЛАЮ ЕЕ РЕШЕЙПИНГ
####################		Convert Image data(string) into Array
# str_to_list = image[0].split(' ')
# print('\n')
# print(str_to_list)
count = 0 
watch = set()
# for img in image:
# 	watch.add(img)


# НУЖНО НАЙТИ дыры в исходных данных

for img in image:
	# print('\t\t',  count, '\n')
	str_to_list = img.split(' ')
	img  		= np.array(str_to_list)
	# img 		= img.astype('float32')
	for i in img:
		watch.add(i)
	# img 		= float(img)
	# print(img)
	# img  		= img.reshape(-1, 96, 96)
	# image[count] = img
	# print(img)
	# for i in img:
		# print(i)
		# print(type(i))
		# i = int(i)/255.0
	# print(str_to_list)
	count += 1
print(watch)



# РАБОЧИЙ ВАРИАНТ
# for img in image:
# 	# print('\t\t',  count, '\n')
# 	str_to_list = img.split(' ')
# 	img  		= np.array(str_to_list)
# 	img 		= img.astype('float32')/255
# 	# print(img)
# 	img  		= img.reshape(-1, 96, 96)
# 	image[count] = img
# 	# print(img)
# 	# for i in img:
# 		# print(i)
# 		# print(type(i))
# 		# i = int(i)/255.0
# 	# print(str_to_list)
# 	count += 1

# print(image)



#  ОТОБРАЗИТЬ ФОТО



# 	СОЗДАТЬ СЕТЬ




















