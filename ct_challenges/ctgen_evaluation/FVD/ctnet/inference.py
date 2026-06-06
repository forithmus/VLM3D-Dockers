#main.py

import timeit

from run_experiment import DukeCTModel
from models import custom_models_ctnet, custom_models_alternative, custom_models_ablation
from load_dataset import dataset_nii
#Note that here NUM_EPOCHS is set to 2 for the purposes of quickly demonstrating
#the code on the fake data. In all of the experiments reported in the paper,
#NUM_EPOCHS was set to 100. No model actually trained all the way to 100 epochs
#due to use of early stopping.
NUM_EPOCHS = 100

if __name__=='__main__':
    ####################################
    # CTNet-83 Model on Whole Data Set #----------------------------------------
    ####################################
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor = 'CTNet28_ctclip_whole_data_18classes',
                custom_net = custom_models_ctnet.CTNetModel,
                custom_net_args = {'n_outputs':18},
                loss = 'bce', loss_args = {},
                num_epochs=NUM_EPOCHS, patience = 15,
                batch_size = 1, device = 'all', data_parallel = True,
                use_test_set = True, task = 'predict_on_test',
                old_params_dir = 'trained_params',
                dataset_class = dataset_nii.CTDataset_2019_10,
                output_dir = "output_dir_new_nii",
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all',
                                    'num_channels':1,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'test':''},
                                    'data_folder':"/shares/menze.dqbm.uzh/ihamam/ctrate_push/CT-RATE/dataset/valid/",
                                    'labels_file': "/home/ihamam/data/23_09_2023_radchest_dataloader/ct-net-models/valid_predicted_labels.csv",
                                    "metadata_file": "/home/ihamam/data/maxpool_ctclip/CT-CLIP/scripts/validation_metadata.csv"})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')

