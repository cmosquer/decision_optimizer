import albumentations as albu

try:
    import torch
    from albumentations.pytorch.transforms import ToTensor
    from . import Albunet
except:
    pass

try:
    from tensorflow.keras.models import model_from_json
except:
    pass

import os, pickle
import cv2
import tqdm as tqdm
import pydicom as pyd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decision_optimizer.utils import automatic_brightness_and_contrast, get_normalized_alpha

class Segmentator():
    def __init__(self, model_config_dict, device='cpu', channels=3, save_jpg_dir=None):
        self.cxr_dir = None
        self.output_masks_dir = None
        self.save_jpg_dir = save_jpg_dir
        if self.save_jpg_dir is not None:
            os.makedirs(save_jpg_dir, exist_ok=True)

        self.current_filename = None
        self.current_orig_image = None
        self.current_preprocessed_image = None
        self.current_mask = None

        self.model_config_dict = model_config_dict
        self.device = device
        self.model = None
        self.channels = channels



    def generate_all_masks(self, cxr_dir, output_masks_dir):
        assert os.path.exists(cxr_dir)
        assert os.path.exists(output_masks_dir)
        self.cxr_dir = cxr_dir
        self.output_masks_dir = output_masks_dir

        os.makedirs(self.output_masks_dir, exist_ok=True)
        for image_filename in os.listdir(self.cxr_dir):
            try:
                self.current_filename = os.path.join(self.cxr_dir, image_filename)
                self.generate_current_mask()
            except Exception as e:
                print(image_filename, e)

    def generate_current_mask(self):
        self.load_current_image()
        self.preprocess_current_image()
        self.process_image()
        self.save_mask()
        self.save_mask_on_cxr()

    def load_current_image(self):
        ext = os.path.splitext(self.current_filename)[-1]
        if ext in ['.jpg', '.png']:
            self.current_orig_image = cv2.imread(self.current_filename, 1)
        else:
            ds = pyd.read_file(self.current_filename, force=True).pixel_array
            assert ds.shape[0] > 0
            norm_image = cv2.normalize(ds, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
            norm_image = norm_image.astype(np.uint8)
            if self.channels == 3:
                self.current_orig_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)

            else:
                self.current_orig_image = norm_image
            if self.save_jpg_dir is not None:
                filename = os.path.join(self.save_jpg_dir, os.path.basename(self.current_filename))
                if ext in ['.jpg', '.png', '.dcm', '.npy']:
                    filename = filename.replace(ext, '.jpg')
                else:
                    filename = filename + '.jpg'
                cv2.imwrite(filename, self.current_orig_image)

    def save_mask(self):
        savingpath = os.path.join(self.output_masks_dir, os.path.basename(self.current_filename))
        ext = os.path.splitext(savingpath)[-1]
        if ext in ['.jpg', '.png', '.dcm']:
            savingpath = savingpath.replace(ext, '.npy')
        pickle.dump(self.current_mask, open(savingpath, 'wb'))
        #print("Saved mask for {} in {}".format(self.current_filename,savingpath))


    def save_mask_on_cxr(self,color_map='Purples'):
        cmap = plt.get_cmap(color_map, 512)
        cmap = mpl.colors.ListedColormap(cmap(np.linspace(0.5, 1, 256)))
        colored_mask = cmap(np.squeeze(self.current_mask))
        th = np.quantile(self.current_mask, q=0.90)
        alpha = np.where(self.current_mask > th, self.current_mask, 0)
        alpha = get_normalized_alpha(alpha, max_alfa=0.5)
        colored_mask[:, :, -1] = alpha
        colored_mask = cv2.resize(colored_mask, (self.current_orig_image.shape[1],
                                                   self.current_orig_image.shape[0]),
                                       interpolation=cv2.INTER_AREA)
        fig, ax = plt.subplots(1, 1, figsize=(40, 40))
        ax.imshow(self.current_orig_image, cmap='gray')
        ax.imshow(colored_mask)
        ax.set_axis_off()
        savingpath = os.path.join(self.output_masks_dir, os.path.basename(self.current_filename))+'.jpg'
        plt.savefig(savingpath, bbox_inches='tight', pad_inches=0)
        plt.cla()
        plt.clf()
        plt.close(fig)


class TorchSegmentator(Segmentator):
    def load_model(self):
        with torch.no_grad():
            if isinstance(self.model_config_dict, dict):
                model_class = getattr(eval(self.model_config_dict['PY']), self.model_config_dict['CLASS'])
                model = model_class(**self.model_config_dict.get('ARGS', None)).to(self.device)
            model.load_state_dict(torch.load(self.model_config_dict['CHECKPOINT'], map_location=self.device))
            model.eval()

        self.model = model

    def preprocess_current_image(self):
        img = {"image": self.current_orig_image}
        if 'ALBU_TRANSFORM' in self.model_config_dict:
            transform = albu.load(self.model_config_dict['ALBU_TRANSFORM'])
            img = transform(**img)

        img = ToTensor()(**img)
        self.current_preprocessed_image = torch.from_numpy(np.expand_dims(img['image'], 0)).to(self.device)

    def process_image(self):
        self.current_mask = torch.sigmoid(self.model(self.current_preprocessed_image)).detach().cpu().numpy()


class KerasSegmentator(Segmentator):
    def load_model(self):
        if self.model is None:
            with open(self.model_config_dict['JSON']) as json_file:
                json_config = json_file.read()
            model = model_from_json(json_config)
            self.model = model

    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

    def preprocess_current_image(self):
        img = automatic_brightness_and_contrast(self.current_orig_image) / 255.0
        img = cv2.resize(img, (self.model_config_dict['DIM'], self.model_config_dict['DIM']))
        img = np.expand_dims(img, axis=0)

        if len(img.shape) < 4:
            img = np.expand_dims(img, axis=-1)
        self.current_preprocessed_image = img

    def process_image(self):
        self.current_mask = np.zeros(self.current_preprocessed_image.shape)
        for checkpoint_path in self.model_config_dict['CHECKPOINT'].split(','):
            self.load_weights(checkpoint_path)
            new_mask = self.model.predict(self.current_preprocessed_image)
            self.current_mask = self.current_mask + new_mask
        self.current_mask = cv2.resize(np.squeeze(self.current_mask)[:, :, 0],
                                       (self.current_orig_image.shape[0],self.current_orig_image.shape[1]))
        self.current_mask = cv2.filter2D(self.current_mask, -1, np.ones((5, 5), np.float32) / 25)

