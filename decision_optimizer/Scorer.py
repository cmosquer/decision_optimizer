import os, pickle
import numpy as np
import cv2
from csv import writer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from decision_optimizer.utils import MaskDataset, rle2mask, get_normalized_alpha
try:
    import torchvision
    import torch
except:
    print("could import torch")


class BaselineScorer():
    def __init__(self, pneumothorax_masks_dir, binarize_mask_th=None, imgs_dir=None):
        assert os.path.exists(pneumothorax_masks_dir)
        self.pneumothorax_masks_dir = pneumothorax_masks_dir
        self.imgs_dir = imgs_dir
        self.current_filename = None
        self.current_score = None
        self.binarize_mask_th = binarize_mask_th
        self.rows_header = ['SOPInstanceUID','pneumothorax_area']

    def obtain_scores(self, output_csv_path, plot_dir=None, calculate_dice_df=None):
        if calculate_dice_df is not None:
            self.rows_header.append('dice')
        if not os.path.exists(output_csv_path):
            with open(output_csv_path, 'a', newline='') as f:
                w = writer(f)
                w.writerow(self.rows_header)
        valid_filenames = [mask_filename for mask_filename in os.listdir(
                            self.pneumothorax_masks_dir) if 'jpg' not in mask_filename]
        for mask_filename in tqdm(valid_filenames):
            self.current_filename = mask_filename
            current_row = self.obtain_current_score()
            if calculate_dice_df is not None:
                dice = self.calculate_dice_with_gt(calculate_dice_df)
                current_row.append(dice)
            with open(output_csv_path, 'a', newline='') as f:
                w = writer(f)
                w.writerow(current_row)
            if plot_dir is not None and self.imgs_dir is not None:
                current_image_filename = os.path.join(self.imgs_dir, self.current_filename) + '.jpg'
                assert os.path.exists(current_image_filename)
                self.current_orig_image = cv2.imread(current_image_filename, 1)
                self.current_saving_path = os.path.join(plot_dir, os.path.basename(self.current_filename))+'.jpg'
                self.save_mask_on_cxr()


    def load_current_pneumothorax_mask(self):
        filename = os.path.join(self.pneumothorax_masks_dir, self.current_filename)
        with open(filename, 'rb') as f:
            pneumothorax_mask = pickle.load(f)
        if self.binarize_mask_th is not None:
            pneumothorax_mask = np.where(pneumothorax_mask > self.binarize_mask_th, 255, 0)
        else:
            pneumothorax_mask = 255*pneumothorax_mask
        self.current_mask = pneumothorax_mask

    def save_mask_on_cxr(self):
        fig, ax = plt.subplots(1, 1, figsize=(40, 40))
        ax.imshow(self.current_orig_image, cmap='gray')
        color_maps = ['Purples']
        masks = [self.current_mask]
        if hasattr(self, 'mask_gt'):
            color_maps.append('Oranges')
            masks.append(self.mask_gt)
            print('plotting gt mask')
        if hasattr(self, 'current_lung_mask'):
            color_maps.append('Greens')
            masks.append(self.current_lung_mask)
        for color_map, mask in zip(color_maps, masks):
            cmap = plt.get_cmap(color_map, 512)
            cmap = mpl.colors.ListedColormap(cmap(np.linspace(0.5, 1, 256)))
            colored_mask = cmap(np.squeeze(mask))
            alpha = get_normalized_alpha(mask, max_alfa=0.5)
            colored_mask[:, :, -1] = alpha
            colored_mask = cv2.resize(colored_mask, (self.current_orig_image.shape[1],
                                                       self.current_orig_image.shape[0]),
                                           interpolation=cv2.INTER_AREA)
            ax.imshow(colored_mask)
        ax.set_axis_off()
        plt.savefig(self.current_saving_path, bbox_inches='tight', pad_inches=0)
        print('Saving img and masks to ', self.current_saving_path)
        plt.cla()
        plt.clf()
        plt.close(fig)

    def obtain_current_score(self):
        self.load_current_pneumothorax_mask()
        self.current_score = np.sum(self.current_mask)
        return [self.current_filename.replace('.jpg', '').replace('.npy', ''),
                            self.current_score]

    def calculate_dice_with_gt(self, calculate_dice_df):
        rle_gt_mask = str(calculate_dice_df[calculate_dice_df.index == self.current_filename]['EncodedPixels'].values[0])
        if rle_gt_mask != '-1':
            self.mask_gt = rle2mask(rle_gt_mask, self.current_mask.shape[2], self.current_mask.shape[3])
        else:
            self.mask_gt = np.zeros((self.current_mask.shape[2], self.current_mask.shape[3]),
                                    dtype=self.current_mask.dtype)
        mask_gt = (self.mask_gt / 255).astype(int)
        mask_pred = (self.current_mask / 255).astype(int)
        areas_sum = mask_gt.sum() + mask_pred.sum()
        if areas_sum == 0:
            return 1.
        areas_intersect = (mask_gt & mask_pred).sum()
        self.mask_gt = np.expand_dims(np.expand_dims(self.mask_gt, axis=0), axis=0)
        return 2 * areas_intersect / areas_sum

class LungMaskScorer(BaselineScorer):
    def __init__(self, lung_masks_dir,
                 pneumothorax_masks_dir, binarize_mask_th=None, imgs_dir=None,
                 lung_mask_type='binary_image'):

        assert os.path.exists(lung_masks_dir)
        assert lung_mask_type in ['binary_image', 'pickle']
        self.lung_mask_type = lung_mask_type
        self.lung_masks_dir = lung_masks_dir
        BaselineScorer.__init__(self,pneumothorax_masks_dir, binarize_mask_th, imgs_dir)
        self.rows_header = ['SOPInstanceUID', 'pneumothorax_area', 'lung_area', 'lung_normalized_score']

    def load_current_lung_mask(self):
        if self.lung_mask_type == 'pickle':
            lung_mask = pickle.load(
                open(os.path.join(self.lung_masks_dir, self.current_filename), 'rb'))
        elif self.lung_mask_type:
            filename = self.current_filename.replace('.npy', '.jpg')
            if '.jpg' not in filename:
                filename = filename + '.jpg'
            lung_mask = cv2.imread(os.path.join(self.lung_masks_dir, filename), cv2.IMREAD_GRAYSCALE).astype(np.int)
        self.current_lung_mask = lung_mask

    def obtain_current_score(self):
        self.load_current_pneumothorax_mask()
        self.current_score = np.sum(self.current_mask)
        self.load_current_lung_mask()
        self.current_normalized_score = self.current_score / np.sum(self.current_lung_mask)
        assert 0 <= self.current_normalized_score <= 1
        return [self.current_filename.replace('.jpg', '').replace('.npy', ''),
                self.current_score,
                np.sum(self.current_lung_mask),
                self.current_normalized_score]

class NormalizationScorer(BaselineScorer):
    def __init__(self, baseline_scores_file, pneumothorax_masks_dir,
                 binarize_mask_th=None, imgs_dir=None, baseline_score_name='pneumothorax_area'
                 ):

        BaselineScorer.__init__(self, pneumothorax_masks_dir, binarize_mask_th, imgs_dir)
        if not os.path.exists(baseline_scores_file):
            "Making baseline scores"
            baseline_scorer = BaselineScorer(pneumothorax_masks_dir, binarize_mask_th, imgs_dir)
            baseline_scorer.obtain_scores(baseline_scores_file)
        self.baseline_scores_file = baseline_scores_file
        self.rows_header = ['SOPInstanceUID', baseline_score_name, 'range_normalized_score']
        self.baseline_score_name = baseline_score_name

    def obtain_range(self, train_sops=None):
        df = pd.read_csv(self.baseline_scores_file).set_index('SOPInstanceUID')
        if train_sops is not None:
            df = df.loc[train_sops]#.reset_index(drop=True)
        self.range_max = np.max(df[self.baseline_score_name])
        self.range_min = np.min(df[self.baseline_score_name])
        print("MAX: ", self.range_max, "MIN: ", self.range_min)

    def obtain_current_score(self):
        df = pd.read_csv(self.baseline_scores_file).set_index('SOPInstanceUID')
        sop = self.current_filename.replace('.jpg', '').replace('.npy', '')
        if sop in list(df.index.values):
            self.current_score = df[df.index == sop][self.baseline_score_name].values[0]
        else:
            if self.baseline_score_name == 'pneumothorax_area':
                self.load_current_pneumothorax_mask()
                self.current_score = np.sum(self.current_mask)
            else:
                print('the baseline score name was not found in df')

        self.current_normalized_score = (self.current_score - self.range_min) / (self.range_max - self.range_min)

        self.current_normalized_score = min(self.current_normalized_score, 1.)
        self.current_normalized_score = max(self.current_normalized_score, 0.)
        try:
            assert 0. <= self.current_normalized_score <= 1.
        except AssertionError:
            print('assertion error ',self.current_normalized_score)
        return [sop,
                self.current_score,
                self.current_normalized_score]


class CNNScorer():

    def __init__(self, pneumothorax_masks_dir, jpgs_dir, usable_ids=None):
        assert os.path.exists(pneumothorax_masks_dir)
        assert os.path.exists(jpgs_dir)

        self.pneumothorax_masks_dir = pneumothorax_masks_dir
        self.jpgs_dir = jpgs_dir
        self.filenames = usable_ids


        self.make_model()

    def make_model(self):
        vgg_based = torchvision.models.vgg19(pretrained=True)
        # Modificar input size a 2 channels y un solo output continuo

        number_features = vgg_based.classifier[6].in_features
        features = list(vgg_based.classifier.children())[:-1]  # Remove last layer
        features.extend([torch.nn.Sigmoid(number_features, 1)])
        vgg_based.classifier = torch.nn.Sequential(*features)

        self.model = vgg_based

    def data_loader(self, dataset_ids):
        # dataloader open pneumthorax mask and jpg and combine them as one 2-channel input + label=presence of pneumothroax
        dataset = MaskDataset(self.pneumothorax_masks_dir, self.jpgs_dir, filenames=dataset_ids)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                      batch_size=1,  # how many samples per batch?
                                      num_workers=1,  # how many subprocesses to use for data loading? (higher = more)
                                      shuffle=True)
        return dataloader

    def get_loaders(self):
        # with filenames, masks_dir y jpgs_dir, get train and valid dataloaders
        self.dataloaders['train'] = self.data_loader(train_ids)
        self.dataloaders['valid'] = self.data_loader(valid_ids)

    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        since = time.time()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # set model to trainable
            # model.train()

            train_loss = 0

            # Iterate over data.
            for i, data in enumerate(self.dataloaders['train']):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                print('{} Loss: {:.4f}'.format(
                    'train', train_loss / len(self.dataloaders['train'])))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def obtain_scores(self):
        pass
        # predict