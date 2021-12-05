import os
import shutil
import random
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# from OCRdataset import OCRDataset
transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            # transforms.Normalize([0.485, 0.456, .406],[0.229, 0.224, 0.225])
                                            ])

save_path = '/home/Data/CarData/'

data_path = '/home/Data/CarOriginData/'
class_folder = os.listdir(data_path)

folder = save_path+'test/'
img_list = os.listdir(folder)
for img in tqdm(img_list):
    image = Image.open(folder+img)
    image = transform(image)
    if image.shape[0] == 1:
        os.remove(folder+img)
# train_path = save_path+'train/'
# test_path = save_path+'test/'

# for folder in class_folder:
#     img_folder = data_path + folder + '/'

#     img_list = os.listdir(img_folder)
#     random.shuffle(img_list)
#     sum = len(img_list)

#     train_set = img_list[:int(sum*0.8)]
#     test_set = img_list[int(sum*0.8):]

#     for img in train_set:
#         if img.split('.')[-1] == 'gif' or img.split('.')[-1] == 'GIF':
#             continue
#         shutil.move(img_folder+img, train_path+img)
#     for img in test_set:
#         if img.split('.')[-1] == 'gif' or img.split('.')[-1] == 'GIF':
#             continue
#         shutil.move(img_folder+img, test_path+img)
    