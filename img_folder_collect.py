import torch.utils.data as data

from PIL import Image
import os
import os.path
import json
import re
from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, train):
    images = []
    file_list = glob(dir+'/**/*')
    
    for filename in file_list:
        target = re.split(r'/',filename)[-2]
        #print(target)
        #filename = item#'{0}/{1}{2}'.format('images',line,'.jpg')
        item = (filename,int(target))
        images.append(item)
    return images
'''
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for filename in os.listdir(d):
            if is_image_file(filename):
                path = '{0}/{1}'.format(target, filename)
                item = (path, class_to_idx[target])
                images.append(item)

    return images
'''

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    def __init__(self, root, train=True,transform=None, target_transform=None,
                 loader=default_loader):
        #classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, train)

        self.root = root
        self.imgs = imgs
        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path

    def __len__(self):
        return len(self.imgs)
