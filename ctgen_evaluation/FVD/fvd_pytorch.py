from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import tensorflow.compat.v1 as tf
import numpy as np
import torch

from FVD import fvd_pytorch_model
from FVD.util import open_url

from ctnet.models import custom_models_ctnet, custom_models_alternative, custom_models_ablation
from ctnet.load_dataset import utils

@torch.no_grad()
def compute_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, model: str, device: str='cuda') -> float:
    if model =="ctnet":
        check_point = torch.load("/opt/app/FVD/ctnet/trained_params/CTNet28_ctclip_whole_data_18classes")

        custom_net_args = {'n_outputs':18},
        print("anan")
        model = custom_models_ctnet.CTNetModel(n_outputs=18).eval()

        # Assuming check_point is a dict with key 'params' being a state_dict
        old_state_dict = check_point['params']
        new_state_dict = {}

        for k, v in old_state_dict.items():
            new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
            new_state_dict[new_key] = v

        check_point['params'] = new_state_dict

        model.load_state_dict(check_point['params'])
        model = model.to(device)

        #videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
        #videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

        #videos_fake = videos_fake[1]
        #videos_real = videos_real[1]

        videos_real = videos_real * 2000
        videos_fake = videos_fake * 2000

        videos_real = videos_real - 1000
        videos_fake = videos_fake - 1000
        #video_real = utils.prepare_ctvol_2019_10_dataset(videos_real, [-1000,200], False, 1, "single" )
        #video_fake = utils.prepare_ctvol_2019_10_dataset(videos_fake, [-1000,200], False, 1, "single" )
        print(videos_fake.min(), videos_fake.max(), "max before ctnet")

        videos_real = torch.stack([
            utils.prepare_ctvol_2019_10_dataset(v, [-1000,200], False, 1, "single")
            for v in videos_real
        ], dim=0)

        videos_fake = torch.stack([
            utils.prepare_ctvol_2019_10_dataset(v, [-1000,200], False, 1, "single")
            for v in videos_fake
        ], dim=0)



        print(videos_fake.shape, "this is inside ctnet")
        print(videos_fake.max(), "max inside ctnet")
        feats_fake = model(videos_fake.to(device)).cpu().numpy()
        feats_real = model(videos_real.to(device)).cpu().numpy()




    return fvd_pytorch_model.compute_fvd(feats_fake, feats_real)