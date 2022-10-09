import cv2
import numpy as np
try:
    import torchvision
    import torch
except:
    print("could import torch")

def get_normalized_alpha(alphas, max_alfa=0.8,min_alfa=0):
    max_alpha = alphas.max()
    min_alpha = alphas.min()
    if max_alpha - min_alpha != 0:
        alphas_norm = ((max_alfa - min_alfa) * (alphas - min_alpha) / (max_alpha - min_alpha+1e-10)) + min_alfa
    else:
        alphas_norm = alphas
    return alphas_norm

def rle2mask(rle, height, width):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]
    return np.transpose(mask.reshape(width, height))

class MaskDataset(torchvision.datasets.VisionDataset):
    def __init__(self, masks_dir, images_dir, filenames=None):
        self.masks_dir = masks_dir
        self.images_dir = images_dir
        if filenames:
            self.filenames = filenames
        else:
            self.filenames = os.listdir(self.masks_dir)

    def __getitem__(self, index):
        mask_filename = os.path.join(self.masks_dir, self.filenames[index])
        with open(mask_filename, 'rb') as f:
            mask = pickle.load(f)

        image_filename = os.path.join(self.images_dir, self.filenames[index])
        with open(image_filename, 'rb') as f:
            image = pickle.load(f)

        return image, mask

    def __len__(self):
        return len(self.filenames)


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray = image
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    if alpha > 2 or beta < -100:
        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    else:
        auto_result = image

    return auto_result


