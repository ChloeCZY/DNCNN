import torch
import numpy as np
import os
from os import path
from torch.utils.data.dataset import Dataset
import cv2
import PIL.Image as pil_image
from PIL import ImageFilter as pil_filter

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        # if not exit
        #
        os.makedirs(path)
        print (path+' is created successfully')
        return True
    else:
        # if path exists,
        print (path+' exists')
        return False

def bgr2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2bgr(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2BGR)
    return im_rgb

class YcbCrLoader(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = os.listdir(root)

    def __getitem__(self, idx):
        im = cv2.imread(path.join(self.root, self.imgs[idx]))
        ycbcr = bgr2ycbcr(im.astype(np.float32)/255).transpose(2, 0, 1)
        return self.imgs[idx], ycbcr

    def __len__(self):
        return len(self.imgs)

#####

def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))

def get_torch_y(img_path, required_width, required_height):
    # image_height, image_width = 510 default
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    image = pil_image.open(img_path).convert('RGB')  #
    if image.width != required_width or image.height != required_height:
        image = image.resize((required_width, required_height), resample=pil_image.BICUBIC) # hr.size = 512 -> 510
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)
    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)
    return y

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

     #rmse = torch.sqrt(torch.mean((img1 - img2) ** 2))
     #return 20 * torch.log10(1.0 / rmse)

# define a class: AverageMeter to calculate the mean value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def downsample(im, down_factor, LRHQ_dir, im_name): # image should be in numpy, which means using cv2 to read image
    LRHQ_path = os.path.join(LRHQ_dir, im_name)
    height = np.size(im,0)
    width = np.size(im,1)
    img_blur = cv2.GaussianBlur(im, (5, 5), 0)
    im_downsample = cv2.resize(img_blur, (height//down_factor, width//down_factor))
    cv2.imwrite(LRHQ_path, im_downsample)
    return im_downsample

def downsample2(): # using the pil package
    pil_filter

def jpeg(im_downsample, JPEG_factor, LRLQ_dir, im_name):
    LRLQ_path = os.path.join(LRLQ_dir, im_name.replace('.bmp','.jpg'))
    cv2.imwrite(LRLQ_path, im_downsample, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_factor])
    im_jpeg = cv2.imread(LRLQ_path)
    return im_jpeg

def interpolation(im_jpeg, down_factor):
    height = np.size(im_jpeg,0)
    width = np.size(im_jpeg,1)
    im_interpolation = cv2.resize(im_jpeg, (int(height*down_factor), int(width*down_factor)), interpolation=cv2.INTER_CUBIC)
    return im_interpolation