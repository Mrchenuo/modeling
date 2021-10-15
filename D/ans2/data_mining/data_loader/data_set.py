import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from cv2 import cv2
from torch.utils.data import Dataset
import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.TIF',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# Note that, the sort() can change the label index
# using sort(): {'generated': 0, 'nature': 1}
# using sort(key=len): {'CG': 0, 'PG': 1}
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort(key=len)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # print(classes)
    # print(class_to_idx)
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    num_in_class = []  # the number of samples in each class
    images_txt = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir), key=len):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            num = 0
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    images_txt.append(target + '/' + fname)
                    num += 1
            num_in_class.append(num)

    return images, num_in_class, images_txt


def pil_loader(path, imgresize=None, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if mode == 'L':
            img = img.convert('L')  # convert image to grey
        elif mode == 'RGB':
            img = img.convert('RGB')  # convert image to rgb image
        elif mode == 'HSV':
            img = img.convert('HSV')
            # elif mode == 'LAB':
            #     return RGB2Lab(img)
        if imgresize is not None:
            img = img.resize(imgresize)
        return img


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def opencv_loader(path, imgresize=None, colorSpace='YCrCb'):
    image = cv2.imread(str(path))
    if colorSpace == "YCrCb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif colorSpace == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if imgresize is not None:
        image = cv2.resize(image, imgresize)
    return image


def default_loader(path, backend, colorSapce, imgresize=None):
    from torchvision import get_image_backend
    if backend == 'accimage':
        return accimage_loader(path)
    elif backend == 'opencv':
        return opencv_loader(path, imgresize, colorSapce)
    else:
        return pil_loader(path, imgresize, colorSapce)


class RIDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None, imgresize=None, colorSapce='RGB',
                 loader=default_loader, backend='opencv'):
        classes, class_to_idx = find_classes(root)
        imgs, num_in_class, images_txt = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        # self.mode = args.img_mode
        # self.input_nc = args.input_nc
        self.imgs = imgs
        self.colorSapce = colorSapce
        self.imgresize = imgresize
        self.num_in_class = num_in_class
        self.images_txt = images_txt
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.backend = backend

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.backend, self.colorSapce, self.imgresize)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)