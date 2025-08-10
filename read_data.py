from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

# 指定数据集目录
root_dir = "dataset/train"

# 指定各标签目录，分别获取数据集
defect_dir = "Defective"
defect_dataset = MyData(root_dir, defect_dir)
non_defect_dir = "non_Defective"
non_defect_dataset = MyData(root_dir, non_defect_dir)

# 也可拼接以获取完整数据集
train_dataset = defect_dataset + non_defect_dataset
img, label = train_dataset[0]
img.show()
