from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
image_path = "dataset/test/Defective/IMG_20201114_102203.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image('test_img', img_array, 4, dataformats='HWC')

for i in range(100):
    writer.add_scalar('y=2x', 2*i, i)

writer.close()
