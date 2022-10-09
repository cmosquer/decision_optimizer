from sklearn.model_selection import train_test_split
import random
import numpy as np

class Splitter():

    def __init__(self, split_level='patient', stratify_criteria='GT', random_state=42):

        self.split_level = split_level
        self.stratify_criteria = stratify_criteria
        self.random_state = random_state

        self.current_trainset_filenames = None
        self.current_testset_filenames = None
        self.current_ground_truth_df = None

    def split_df(self, ground_truth_df, train_size=0.5):

        self.current_ground_truth_df = ground_truth_df
        assert self.stratify_criteria in ground_truth_df.columns
        assert self.split_level in ground_truth_df.columns

        split_ids = sorted(list(set(ground_truth_df[self.split_level])))

        gt_ids = [ground_truth_df[ground_truth_df[self.split_level] == split_id][self.stratify_criteria].max() for split_id in split_ids] # si al menos una imagen del apciente tiene GT=1, se considera GT=1 para el stratify split
        split_ids_train, split_ids_test = train_test_split(split_ids, test_size=1-train_size, stratify=gt_ids,
                         random_state=self.random_state)

        self.current_trainset_filenames = list(ground_truth_df[ground_truth_df[self.split_level].isin(split_ids_train)].index.values)
        self.current_testset_filenames = list(ground_truth_df[ground_truth_df[self.split_level].isin(split_ids_test)].index.values)
        self.current_ground_truth_df['split_assignation'] = ['train' if split_id in split_ids_train else 'test' for split_id in self.current_ground_truth_df[self.split_level]]

    def get_test_subset(self, pos_prior, label_column='GT', final_size=None, required_N_positives=None, 
                        ):
        assert self.current_testset_filenames is not None
        self.current_ground_truth_df[label_column] = self.current_ground_truth_df[label_column].astype(int)
        positive_test_samples = self.current_ground_truth_df[(self.current_ground_truth_df.split_assignation == 'test') & (self.current_ground_truth_df[label_column] == 1)]#.reset_index(drop=True)
        negative_test_samples = self.current_ground_truth_df[(self.current_ground_truth_df.split_assignation == 'test') & (self.current_ground_truth_df[label_column] == 0)]#.reset_index(drop=True)
        #print("Total raw positives available for test scenarios: ", len(positive_test_samples))
        if final_size:
            N_positive = int(pos_prior*final_size)
            if N_positive > len(positive_test_samples):
                print(f'Not enough positive samples for pos_prior={pos_prior} and final_size={final_size}')
                final_size = int(len(positive_test_samples) / pos_prior)
                N_positive = len(positive_test_samples)
                print(f"Using final size of {final_size}")
        elif required_N_positives:
            N_positive = required_N_positives
            final_size = int(N_positive / pos_prior)
        else:
            N_positive = len(positive_test_samples)
            final_size = int(N_positive / pos_prior)

        N_negative = final_size - N_positive
        negative_idxs = np.arange(len(negative_test_samples))
        positive_idxs = np.arange(len(positive_test_samples))
        random.seed(self.random_state)
        random.shuffle(negative_idxs)
        random.shuffle(positive_idxs)
        new_test_df = positive_test_samples.iloc[positive_idxs[:N_positive]].append(
            negative_test_samples.iloc[negative_idxs[:N_negative]])#, ignore_index=True)
        return new_test_df