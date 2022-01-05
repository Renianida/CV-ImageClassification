import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

class CustomDataset(Dataset):
	def __init__(self):
		self.imgs_path = 'dataset'
		file_list = glob.glob(self.imgs_path + '/*')
		# print(file_list)

		self.data = []
		for class_path in file_list:
			class_name = class_path.split('/')[-1]
			for img_path in glob.glob(class_path + '/' +'*.jpg'):
				self.data.append([img_path, class_name])

		# print(*self.data, sep='\n')
		self.class_map = {'Bread' : 0, 'Dessert' : 1, 'Meat' : 2, 'Soup' : 3}
		self.img_dim = (512, 512)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]

		# сопоставление имени с числом
		class_id = self.class_map[class_name]

	    # преобразуем целочисленное значение class_id в тензор
	    # также увеличиваем его размерность, ссылаясь на него как [class_id].
	    # Это необходимо для того, чтобы обеспечить возможность пакетной
	    # обработки данных в тех размерах, которые требуются torch.
		class_id = torch.tensor([class_id])

		return img_path, class_id.float()

	def getImgsTensors(self, imgs_path):
		output_tensor = torch.tensor([], dtype=torch.float)

		for img_path in imgs_path:
			img = cv2.imread(img_path)
			# print(img.shape)
			img = cv2.resize(img, self.img_dim)

		    # преобразуем переменные в тензоры (torch.from_numpy позволяет
		    # преобразовать массив numpy в тензор)
			img_tensor = torch.from_numpy(img)
			# замены осей ((Каналы, Ширина, Высота))
			img_tensor = img_tensor.permute(2, 0, 1).float()

			img_tensor = img_tensor.unsqueeze(0)

			output_tensor = torch.cat((output_tensor, img_tensor))

		return output_tensor

	def getName(self, value):
	    for k, v in self.class_map.items():
	        if v == value:
	            return k

if __name__ == "__main__":
	# Чтобы протестировать набор данных и наш загрузчик данных, в главной
	# функции нашего скрипта мы создаем экземпляр созданного
	# CustomDataset и назовем его dataset.
	dataset = CustomDataset()
	dataset_train, dataset_test = train_test_split(dataset, test_size=0.7)

	data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
	data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=True)

	# Display image and label.
	train_features, train_labels = next(iter(data_loader_train))
	train_features = dataset.getImgsTensors(train_features)

	print(f"Feature batch shape: {train_features.size()}")
	print(f"Labels batch shape: {train_labels.size()}")
	img = train_features[0].squeeze()
	label = train_labels[0]

	plt.title(f"Label: {dataset.getName(label)}")
	plt.imshow(cv2.cvtColor(img.permute(1, 2, 0).numpy() / 255, cv2.COLOR_BGR2RGB))
	plt.show()
	# Провели первичную классификацию данных. Теперь нужно разделить данные
	# на тестовые и тренировочные
