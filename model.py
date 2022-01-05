import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 18, 5)
        self.fc1 = nn.Linear(18 * 29 * 29, 60)  # 128px, 29 - зависимость от изображения и MaxPool2d (при 32px - 5)
        self.fc2 = nn.Linear(60, 42)
        self.fc3 = nn.Linear(42, 4)

        print('Model created!')

    def forward(self, x):
        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 18 * 29 * 29)  # 128px, 29 - зависимость от изображения MaxPool2d (при 32px - 5)
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.size())

        return x
