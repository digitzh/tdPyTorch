from PIL import Image
from torchvision import transforms

img_path = "dataset\\test\\Defective\\IMG_20201114_100159.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img.shape)
