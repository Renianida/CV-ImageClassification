import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
		img = cv2.imread(img_path)
		# print(img.shape)
		img = cv2.resize(img, self.img_dim)
	    # сопоставление имени с числом
		class_id = self.class_map[class_name]
	    # преобразуем переменные в тензоры (torch.from_numpy позволяет
	    # преобразовать массив numpy в тензор)
		img_tensor = torch.from_numpy(img)
	    # замены осей ((Каналы, Ширина, Высота))
		img_tensor = img_tensor.permute(2, 0, 1)
	    # преобразуем целочисленное значение class_id в тензор
	    # также увеличиваем его размерность, ссылаясь на него как [class_id].
	    # Это необходимо для того, чтобы обеспечить возможность пакетной
	    # обработки данных в тех размерах, которые требуются torch.
		class_id = torch.tensor([class_id])

		return img_tensor.float(), class_id.float()

if __name__ == "__main__":
	# Чтобы протестировать набор данных и наш загрузчик данных, в главной
	# функции нашего скрипта мы создаем экземпляр созданного
	# CustomDataset и назовем его dataset.
	dataset = CustomDataset()
	dataset_train, dataset_test = train_test_split(dataset, test_size=0.7)

	# 	data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
	data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
	data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=True)
	# 	for imgs, labels in data_loader_train:
	#   		print("Batch of images has shape: ",imgs.shape)
			# print("Batch of labels has shape: ", labels.shape)
	# Display image and label.
	train_features, train_labels = next(iter(data_loader_train))
	print(f"Feature batch shape: {train_features.size()}")
	print(f"Labels batch shape: {train_labels.size()}")
	img = train_features[0].squeeze()
	label = train_labels[0]
	plt.imshow(img, cmap="gray")
	plt.show()
	print(f"Label: {label}")
	# Провели первичную классификацию данных. Теперь нужно разделить данные
	# на тестовые и тренировочные
