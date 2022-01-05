import glob
import cv2
import numpy as np

import pickle

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

from model import ClassificationModel

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
		self.img_dim = (128, 128) # (32, 32)

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

		return img_path, class_id

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

	data_loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
	data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=True)

	# Display image and label.
	train_features, train_labels = next(iter(data_loader_train))
	train_features = dataset.getImgsTensors(train_features)

	print(f"Feature batch shape: {train_features.size()}")
	print(f"Labels batch shape: {train_labels.size()}")

	plt.title('Labels: ' + ', '.join([dataset.getName(i) for i in train_labels]))
	gridImgs = torchvision.utils.make_grid(train_features)
	plt.imshow(cv2.cvtColor(gridImgs.permute(1, 2, 0).numpy() / 255, cv2.COLOR_BGR2RGB))
	plt.show()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	model = ClassificationModel().to(device)
	print(model)

	criterion = nn.CrossEntropyLoss()
	# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	best_loss = np.inf

	for epoch in range(100):
		epoch_loss = 0.0

		for i, data in enumerate(data_loader_train, 0):
			# получаем вводные данные
			inputs, labels = data
			labels = labels.to(device)
			inputs = dataset.getImgsTensors(inputs).to(device)

			# обнуляем параметр gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			# exit()

			loss = criterion(outputs, torch.max(labels, 1)[0])
			loss.backward()

			optimizer.step()

			epoch_loss += loss.item()

		print('[%d] loss: %.10f' % (epoch + 1, epoch_loss))

		if epoch_loss < best_loss:
			best_loss = epoch_loss
			best_model = pickle.loads(pickle.dumps(model))

	print('Тренировка завершена, наименьшая ошибка:', best_loss)

	print('Проверка наилучшей модели')

	best_model.eval()
	epoch_loss = 0

	with torch.no_grad():
		for i, data in enumerate(data_loader_test, 0):
			# получаем вводные данные
			inputs, labels = data
			labels = labels.to(device)
			inputs = dataset.getImgsTensors(inputs).to(device)

			outputs = best_model(inputs)

			loss = criterion(outputs, torch.max(labels, 1)[0])

			epoch_loss += loss.item()

	print('Ошибка на тестовой выборке:', epoch_loss / len(dataset_test))

	# Display image and label.
	test_features, test_labels = next(iter(data_loader_test))
	test_labels = test_labels.to(device)
	test_features = dataset.getImgsTensors(test_features).to(device)

	outputs = best_model(test_features)
	outputs = torch.max(outputs, 1)[1]

	plt.title('Data labels: ' + ', '.join([dataset.getName(i) for i in test_labels]) + \
			  '\nPredicted labels: ' + ', '.join([dataset.getName(i) for i in outputs]))
	gridImgs = torchvision.utils.make_grid(test_features).cpu()
	plt.imshow(cv2.cvtColor(gridImgs.permute(1, 2, 0).numpy() / 255, cv2.COLOR_BGR2RGB))
	plt.show()
