import numpy as np

from torch.utils.data import Dataset


class DatasetReader(Dataset):
    def __init__(self, samples, labels, transforms=None):
        self.samples = samples
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'x': self.samples[item], 'y': self.labels[item]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class LondonDatasetReader(Dataset):
    def __init__(self, samples_list, labels_list, transforms=None):
        min_number_of_samples = min([samples_list[i].shape[0] for i in range(len(samples_list))])

        new_samples_list = [samples_list[i][-min_number_of_samples:] for i in range(len(samples_list))]
        new_labels_list = [labels_list[i][-min_number_of_samples:] for i in range(len(labels_list))]

        self.samples_list = np.array(new_samples_list)
        self.labels_list = np.array(new_labels_list)

        self.transform = transforms

    def __len__(self):
        return self.samples_list.shape[1]

    def __getitem__(self, item):
        samples = {'x': self.samples_list[:, item, :, :], 'y': self.labels_list[:, item, :]}

        if self.transform:
            samples = self.transform(samples)

        return samples


