"""
DnCNN

"""
import argparse
import glob
import h5py
import numpy as np
from PIL import Image, ImageFilter
from utils import convert_rgb_to_y, mkdir
import os


def train(args):
    h5_file = h5py.File(args.h5_path, 'w')

    lr_patches = []
    hr_patches = []

    # for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
    train_list = os.listdir(args.images_dir)
    count = 0
    for num, img_name in enumerate(train_list):
        img_jpgname = img_name.replace('.bmp','.jpg')
        image_path = os.path.join(args.images_dir, img_name)
        image_jpgpath = os.path.join(args.jpg_image_dir, img_jpgname)
        hr = Image.open(image_path).convert('RGB')
        # hr_width = (hr.width // args.scale) * args.scale
        # hr_height = (hr.height // args.scale) * args.scale
        # hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC) # hr.size = 512 -> 510
        # hr_blur = hr.filter(ImageFilter.GaussianBlur(2))
        # lr = hr_blur.resize((hr_width // args.scale, hr_height // args.scale), resample=Image.BICUBIC) # lr.size = 510/3 -> 170

        hr.save(image_jpgpath, quality=args.JPEG_factor)
        img_pil_jpg = Image.open(image_jpgpath).convert('RGB')
        # TODO: add denoise.
        hr = np.array(hr).astype(np.float32)
        lr = np.array(img_pil_jpg).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        # cutting the pairs of patches

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])
                count = count + 1

    print('number of pairs: ', count)
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()

# compare eval() and train() why they are different
def eval(args):
    h5_file = h5py.File(args.h5_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    # how to get IR image: downsample the HR image (bicubic resize)
    # what to input upsample the IR image, which gets IR'
    # chanel convert: rgb to y
    # no patch the image

    eval_list = os.listdir(args.images_dir)
    count = 0
    for num, img_name in enumerate(eval_list):
        img_jpgname = img_name.replace('.bmp','.jpg')
        image_path = os.path.join(args.images_dir, img_name)
        image_jpgpath = os.path.join(args.jpg_image_dir, img_jpgname)
        hr = Image.open(image_path).convert('RGB')

        hr.save(image_jpgpath, quality=args.JPEG_factor)
        img_pil_jpg = Image.open(image_jpgpath).convert('RGB')
        # TODO: add the denoise.

        hr = np.array(hr).astype(np.float32)
        lr = np.array(img_pil_jpg).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        # cutting the pairs of patches
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                # lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                # hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])
                lr_group.create_dataset(str(count), data=lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_group.create_dataset(str(count), data=hr[i:i + args.patch_size, j:j + args.patch_size])
                count = count+1
    print(count)

    h5_file.close()

# input the dir of the ground true 512*512 bmp image with JPEG-factor and SR scale
# output the images that are blurred, downsampled and denoised, stored in another jpg_dir
def test(args):
    test_list = os.listdir(args.images_dir)
    for num, img_name in enumerate(test_list):
        img_jpgname = img_name.replace('.bmp','.jpg')
        image_path = os.path.join(args.images_dir, img_name)
        image_jpgpath = os.path.join(args.jpg_image_dir, img_jpgname)
        hr = Image.open(image_path).convert('RGB')
        # hr_width = (hr.width // args.scale) * args.scale
        # hr_height = (hr.height // args.scale) * args.scale
        #hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC) # hr.size = 512 -> 510
        hr_blur = hr.filter(ImageFilter.GaussianBlur(2))
        lr = hr_blur.resize((hr.width // args.scale, hr.height // args.scale), resample=Image.BICUBIC) # lr.size = 512/3 -> 170
        lr.save(image_jpgpath, quality=args.JPEG_factor)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default=r'E:\PythonCode\bishe\baseline2\database\test\test')
    parser.add_argument('--h5-path', type=str, default=r'E:\PythonCode\bishe\baseline2\h5')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--function', type=str, default='test')
    parser.add_argument('--eval', action='store_true', default=False)  # input --eval, eval(); not, train()
    parser.add_argument('--JPEG-factor', type=int, default=40)
    parser.add_argument('--jpg-image-dir', type=str, default=(os.path.join(os.getcwd(),'database')))
    args = parser.parse_args()
    args.jpg_image_dir = os.path.join(args.jpg_image_dir, args.function)
    args.jpg_image_dir = os.path.join(args.jpg_image_dir, ('jpg_image_'+args.function))
    mkdir(args.images_dir), mkdir(args.jpg_image_dir)
    action = args.function
    print(args)

    if action == 'train':
        args.h5_path = os.path.join(args.h5_path, 'train_'+str(args.scale)+'.h5')  # input h5
        train(args)
        print('train')
    elif action == 'eval':
        args.h5_path = os.path.join(args.h5_path, 'eval_' + str(args.scale) + '.h5')  # input h5
        eval(args)
        print('eval')
    elif action == 'test':
        test(args)
        print('test')
    else:
        print('please enter train, eval or test in --function')

