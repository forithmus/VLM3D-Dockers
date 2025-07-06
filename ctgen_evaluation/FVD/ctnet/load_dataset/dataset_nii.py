#custom_datasets.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import os
import pickle
import numpy as np
import pandas as pd
import glob
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from . import utils

import nibabel as nib

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

###################################################
# PACE Dataset for Data Stored in 2019-10-BigData #-----------------------------
###################################################
class CTDataset_2019_10(Dataset):
    def __init__(self, setname, label_type_ld,
                 label_meanings, num_channels, pixel_bounds,
                 data_augment, crop_type,
                 selected_note_acc_files, data_folder="", labels_file="", metadata_file=""):
        """CT Dataset class that works for preprocessed data in 2019-10-BigData.
        A single example (for crop_type == 'single') is a 4D CT volume:
            if num_channels == 3, shape [134,3,420,420]
            if num_channels == 1, shape [402,420,420]

        Variables:
        <setname> is either 'train' or 'valid' or 'test'
        <label_type_ld> is 'disease_new'
        <label_meanings>: list of strings indicating which labels should
            be kept. Alternatively, can be the string 'all' in which case
            all labels are kept.
        <num_channels>: number of channels to reshape the image to.
            == 3 if the model uses a pretrained feature extractor.
            == 1 if the model uses only 3D convolutions.
        <pixel_bounds>: list of ints e.g. [-1000,200]
            Determines the lower bound, upper bound of pixel value clipping
            and normalization.
        <data_augment>: if True, perform data augmentation.
        <crop_type>: is 'single' for an example consisting of one 4D numpy array
        <selected_note_acc_files>: This should be a dictionary
            with key equal to setname and value that is a string. If the value
            is a path to a file, the file must be a CSV. Only note accessions
            in this file will be used. If the value is not a valid file path,
            all available note accs will be used, i.e. the model will be
            trained on the whole dataset."""
        self.setname = setname
        self.define_subsets_list()
        self.paths=[]
        self.data_folder=data_folder
        self.metadata_file = metadata_file
        self.labels_file = labels_file
        self.label_type_ld = label_type_ld
        self.label_meanings = label_meanings
        self.num_channels = num_channels
        self.pixel_bounds = pixel_bounds
        if self.setname == 'train':
            self.data_augment = data_augment
        else:
            self.data_augment = False
        print('For dataset',self.setname,'data_augment is',self.data_augment)
        self.crop_type = crop_type
        assert self.crop_type == 'single'
        self.selected_note_acc_files = selected_note_acc_files
        self.get_file_locs()

        #Define location of the CT volumes
        #self.volume_log_df = pd.read_csv('/home/ihamam/data/23_09_2023_radchest_dataloader/inferred.csv',header=0,index_col=0)

        #Get the example ids

        #Get the ground truth labels
        self.labels_df = self.get_labels_df()

    # Pytorch Required Methods #------------------------------------------------
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """Return a single sample at index <idx>. The sample is a Python
        dictionary with keys 'data' and 'gr_truth' for the image and label,
        respectively"""
        return self._get_pace(self.paths[idx])

    def get_file_locs(self):

        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))
            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

                for nii_file in nii_files:
                    self.paths.append(nii_file)



    # Ground Truth Label Methods #----------------------------------------------
    def get_labels_df(self):
        #Get the ground truth labels based on requested label type.

        labels_df = pd.read_csv(self.labels_file, header=0, index_col = 0)

        #Now filter the ground truth labels based on the desired label meanings:
        if self.label_meanings != 'all': #i.e. if you want to filter
            labels_df = labels_df[self.label_meanings]
        return labels_df

    def resize_array(self, array, current_spacing, target_spacing):

        # Calculate new dimensions
        original_shape = array.shape[2:]
        scaling_factors = [
            current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
        ]
        new_shape = [
            int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
        ]
        # Resize the array
        resized_array=F.interpolate(array, size=new_shape, mode='trilinear',align_corners=False)

        return resized_array

    # Fetch a CT Volume (__getitem__ implementation) #--------------------------
    def _get_pace(self, path):
        #Load compressed npz file: [slices, square, square]
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        df = pd.read_csv(self.metadata_file) #select the metadata
        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        img_data = slope * img_data + intercept

        img_data = img_data.transpose(2, 0, 1)
        print(img_data.shape)

        current = (z_spacing, xy_spacing, xy_spacing)

        target = (target_z_spacing,target_x_spacing, target_y_spacing)

        tensor = torch.tensor(img_data, dtype=torch.float32)
        tensor_new = self.resize_array(tensor.unsqueeze(0).unsqueeze(0), current, target)
        tensor = tensor_new.squeeze(0).squeeze(0)


        #Prepare the CT volume data (already torch Tensors)
        data = utils.prepare_ctvol_2019_10_dataset(tensor.cpu().detach().numpy(), self.pixel_bounds, self.data_augment, self.num_channels, self.crop_type)
        #Get the ground truth:
        #note_acc = self.volume_log_df[self.volume_log_df['full_filename_npz']==volume_acc].index.values.tolist()[0]
        print(data.shape)
        volume_acc = path.split("/")[-1]
        gr_truth = self.labels_df.loc[volume_acc].values

        gr_truth = torch.from_numpy(gr_truth).squeeze().type(torch.float)

        #When training on only one abnormality you must unsqueeze to prevent
        #a dimensions error when training the model:
        if len(self.label_meanings)==1:
            gr_truth = gr_truth.unsqueeze(0)

        #Create the sample
        sample = {'data': data, 'gr_truth': gr_truth, 'volume_acc': volume_acc}
        return sample

    # Sanity Check #------------------------------------------------------------
    def define_subsets_list(self):
        assert self.setname in ['train','valid','test']
        if self.setname == 'train':
            self.subsets_list = ['imgtrain']
        elif self.setname == 'valid':
            self.subsets_list = ['imgvalid_a']
        elif self.setname == 'test':
            self.subsets_list = ['imgtest_a','imgtest_b','imgtest_c','imgtest_d']
        print('Creating',self.setname,'dataset with subsets',self.subsets_list)
