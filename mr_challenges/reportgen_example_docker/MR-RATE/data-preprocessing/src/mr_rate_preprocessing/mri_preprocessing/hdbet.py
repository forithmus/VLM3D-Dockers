"""
MR-RATE MRI Preprocessing Pipeline - HD-BET Brain Segmentation


HD-BET and BrainLesion Suite Attribution
----------------------------------------
BrainSegmentor in this script is implemented based on the `run_hd_bet`, `predict_case_3D_net`, 'save_segmentation_nifti' functions from the HD-BET repository:
https://github.com/BrainLesion/HD-BET/blob/main/brainles_hd_bet/run.py
https://github.com/BrainLesion/HD-BET/blob/main/brainles_hd_bet/predict_case.py
https://github.com/BrainLesion/HD-BET/blob/main/brainles_hd_bet/data_loading.py

If you are using this script that uses HD-BET and BrainLesion Suite, please cite the following publications:

Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A,
Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P.
Automated brain extraction of multi-sequence MRI using artificial neural networks.
Hum Brain Mapp. 2019; 1-13. https://doi.org/10.1002/hbm.24750

Kofler, F., Rosier, M., Astaraki, M., Möller, H., Mekki, I. I., Buchner, J. A., Schmick, 
A., Pfiffer, A., Oswald, E., Zimmer, L., Rosa, E. de la, Pati, S., Canisius, J., Piffer, 
A., Baid, U., Valizadeh, M., Linardos, A., Peeken, J. C., Shit, S., … Menze, B. (2025). 
BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image 
Analysis https://arxiv.org/abs/2507.09036
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import SimpleITK as sitk

# HD-BET imports
from brainles_hd_bet.config import HD_BET_Config
from brainles_hd_bet.data_loading import load_and_preprocess, resize_segmentation
from brainles_hd_bet.predict_case import pad_patient_3D
from brainles_hd_bet.utils import (
    get_params_fname,
    maybe_download_parameters,
    postprocess_prediction,
)


class BrainSegmentor:
    """
    Adapted from the `run_hd_bet` and `predict_case_3D_net` functions from the HD-BET repository:
    https://github.com/BrainLesion/HD-BET/blob/main/brainles_hd_bet/run.py
    https://github.com/BrainLesion/HD-BET/blob/main/brainles_hd_bet/predict_case.py
    

    Brain segmentation using HD-BET with optimized GPU inference.
    
    Variations from the original implementation: Loads models once during initialization 
    and keeps them in GPU memory for efficient inference. Supports multi-gpu data-parallel 
    inference, TTA in GPU, batched infernence for TTA, reusing inputs in ensembling, 
    mixed precision inference, and torch.compile() for faster processing.
    
    Attributes:
        mode: 'fast' (1 model) or 'accurate' (5 models ensemble)
        device: GPU device ID or 'cpu'
        do_tta: Whether to perform test-time augmentation (mirroring)
        postprocess: Whether to keep only largest connected component
        compile: Whether to use torch.compile() for faster inference
        mixed_prec: Whether to use BF16 mixed precision inference
        config: HD_BET_Config instance with network and inference parameters
        networks: List of pre-loaded networks ready for inference
    """
    
    # TTA flip configurations for 5D tensor [B, C, D, H, W] where D=2, H=3, W=4
    TTA_FLIP_CONFIGS = [
        [],        # m=0: no flip
        [4],       # m=1: flip W
        [3],       # m=2: flip H
        [3, 4],    # m=3: flip H, W
        [2],       # m=4: flip D
        [2, 4],    # m=5: flip D, W
        [2, 3],    # m=6: flip D, H
        [2, 3, 4], # m=7: flip D, H, W
    ]

    @staticmethod
    def save_segmentation_nifti(segmentation, dct, out_fname, order=1):
        """
        This function is copied from the brainles_hd_bet.data_loading.save_segmentation_nifti function 
        to change the output dtype from np.int32 to np.uint8:
        https://github.com/BrainLesion/HD-BET/blob/main/brainles_hd_bet/data_loading.py


        segmentation must have the same spacing as the original nifti (for now). segmentation may have been cropped out
        of the original image

        dct:
        size_before_cropping
        brain_bbox
        size -> this is the original size of the dataset, if the image was not resampled, this is the same as size_before_cropping
        spacing
        origin
        direction

        :param segmentation:
        :param dct:
        :param out_fname:
        :return:
        """
        old_size = dct.get("size_before_cropping")
        bbox = dct.get("brain_bbox")
        if bbox is not None:
            seg_old_size = np.zeros(old_size)
            for c in range(3):
                bbox[c][1] = np.min((bbox[c][0] + segmentation.shape[c], old_size[c]))
            seg_old_size[
                bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
            ] = segmentation
        else:
            seg_old_size = segmentation
        if np.any(np.array(seg_old_size.shape) != np.array(dct["size"])[[2, 1, 0]]):
            seg_old_spacing = resize_segmentation(
                seg_old_size, np.array(dct["size"])[[2, 1, 0]], order=order
            )
        else:
            seg_old_spacing = seg_old_size
        seg_resized_itk = sitk.GetImageFromArray(seg_old_spacing.astype(np.uint8)) # np.int32 was used originally
        seg_resized_itk.SetSpacing(np.array(dct["spacing"])[[0, 1, 2]])
        seg_resized_itk.SetOrigin(dct["origin"])
        seg_resized_itk.SetDirection(dct["direction"])
        sitk.WriteImage(seg_resized_itk, out_fname)
    
    def __init__(
        self,
        mode: str = "accurate",
        device: int | str = 0,
        do_tta: bool = True,
        postprocess: bool = False,
        compile: bool = False,
        mixed_prec: bool = False,
    ):
        """
        Initialize the brain segmentor by loading models into GPU memory.
        
        Args:
            mode: 'fast' (1 model) or 'accurate' (5 models ensemble)
            device: GPU device ID (int) or 'cpu'
            do_tta: Enable test-time augmentation via mirroring
            postprocess: Keep only largest connected component in output
            compile: Enable torch.compile() for faster inference
            mixed_prec: Enable BF16 mixed precision
        """
        self.mode = mode
        self.device = device
        self.do_tta = do_tta
        self.postprocess = postprocess
        self.compile = compile
        self.mixed_prec = mixed_prec
        
        # Raise error if CUDA not available and device is GPU
        if not torch.cuda.is_available() and device != "cpu":
            raise Exception(f"Selected device={device} is GPU but CUDA is not available")
        
        # Initialize HD-BET config (contains network params and inference settings)
        self.config = HD_BET_Config()
        
        # Download model parameter files and create pre-loaded networks
        self.networks = self._create_and_load_networks()

    def _create_and_load_networks(self) -> List:
        """
        Download parameters, create networks, and move to device.
        
        Returns:
            List of networks with pre-loaded weights ready for inference
        """
        if self.mode == "fast":
            model_indices = [0]
        elif self.mode == "accurate":
            model_indices = list(range(5))
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. Expected: 'fast' or 'accurate'"
            )

        networks = []
        
        for i in model_indices:
            # Download parameters if needed
            maybe_download_parameters(i)
            params_file = get_params_fname(i)
            
            # Verify parameter file exists
            if not params_file.exists():
                raise FileNotFoundError(f"Parameter file not found: {params_file}")
            
            # Create network with weights loaded and eval mode set
            # get_network(train=False) sets: net.train(False), net.apply(SetNetworkToVal(...)), net.do_ds=False
            net, _ = self.config.get_network(
                train=False, 
                pretrained_weights=str(params_file)
            )
            
            # Move to device
            if self.device == "cpu":
                net = net.cpu()
            else:
                net = net.cuda(self.device)
            
            # Apply torch.compile for GPU if requested
            if self.device != "cpu" and self.compile:
                net = torch.compile(net)
            
            networks.append(net)
        
        return networks
    
    def _run_inference(self, patient_data: np.ndarray) -> np.ndarray:
        """
        Run ensemble inference with optional batched TTA.
        
        Prepares data once, runs all networks, and returns the final segmentation.
        When TTA is enabled, all 8 mirror augmentations are processed in a single
        batched forward pass per network. Flipping, averaging and argmax done on GPU.
        
        Args:
            patient_data: Input data array of shape (C, D, H, W)
            
        Returns:
            Segmentation array of shape (D, H, W) with integer labels
        """
        with torch.inference_mode():
            # Pad input to be divisible by network requirements
            pad_res = []
            for i in range(patient_data.shape[0]):
                t, old_shape = pad_patient_3D(
                    patient_data[i], 
                    self.config.net_input_must_be_divisible_by, 
                    self.config.val_min_size
                )
                pad_res.append(t[None])
            
            patient_data_padded = np.vstack(pad_res)
            
            # Create input tensor: (1, C, D, H, W)
            data = np.zeros(tuple([1] + list(patient_data_padded.shape)), dtype=np.float32)
            data[0] = patient_data_padded
            
            # Move to GPU once
            if self.device == "cpu":
                data_tensor = torch.from_numpy(data).float()
            else:
                data_tensor = torch.from_numpy(data).float().cuda(self.device)
            
            # Prepare TTA variants once (reused for all networks)
            if self.do_tta:
                # Determine which TTA configs to use based on mirror_axes
                active_configs = [
                    flip_dims for flip_dims in self.TTA_FLIP_CONFIGS
                    if all(dim in self.config.da_mirror_axes for dim in flip_dims)
                ]
                
                # Generate all TTA variants on GPU using torch.flip
                variants = []
                for flip_dims in active_configs:
                    if len(flip_dims) == 0:
                        variants.append(data_tensor)
                    else:
                        variants.append(torch.flip(data_tensor, dims=flip_dims))
                
                # Stack into batch: (num_variants, C, D, H, W)
                batched = torch.cat(variants, dim=0)
            else:
                # No TTA: single input
                batched = data_tensor
                active_configs = [[]]  # Single "no flip" config
            
            # Collect predictions from all networks
            all_preds = []
            
            for net in self.networks:
                # Forward pass with optional mixed precision
                if self.mixed_prec and self.device != "cpu":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = net(batched)
                else:
                    outputs = net(batched)
                
                if self.do_tta:
                    # Un-flip outputs on GPU
                    unflipped = []
                    for i, flip_dims in enumerate(active_configs):
                        out = outputs[i:i+1]
                        if len(flip_dims) > 0:
                            out = torch.flip(out, dims=flip_dims)
                        unflipped.append(out)
                    
                    # Average TTA predictions for this network
                    stacked = torch.cat(unflipped, dim=0)
                    net_pred = stacked.mean(dim=0, keepdim=True)
                else:
                    net_pred = outputs
                
                all_preds.append(net_pred)
            
            # Average across all networks on GPU
            ensemble_pred = torch.cat(all_preds, dim=0).mean(dim=0)
            
            # Crop to original shape and compute argmax on GPU
            ensemble_pred = ensemble_pred[:, :old_shape[0], :old_shape[1], :old_shape[2]]
            seg = ensemble_pred.argmax(dim=0)
            
            # Transfer final segmentation to CPU
            return seg.cpu().numpy()
    
    def __call__(
        self,
        input_path: Path | str,
        output_mask_path: Path | str,
    ) -> bool:
        """
        Perform brain segmentation on a single image.
        
        Args:
            input_path: Path to input NIfTI image
            output_mask_path: Path to save output brain mask
            
        Returns:
            True if successful, False otherwise
        """
        input_path = Path(input_path)
        output_mask_path = Path(output_mask_path)
        
        # Ensure output directory exists
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess input image
        data, data_dict = load_and_preprocess(str(input_path))
        
        # Run ensemble inference (data prep done once, all networks processed)
        seg = self._run_inference(data)
        
        # Optionally postprocess (keep largest connected component)
        if self.postprocess:
            seg = postprocess_prediction(seg)
        
        # Save segmentation mask
        self.save_segmentation_nifti(seg, data_dict, str(output_mask_path))
        
        return True


def parse_devices(device_str: str) -> List[int | str]:
    """
    Parse device string into list of device IDs.
    
    Args:
        device_str: Device specification (e.g., '0', '0,1,2', 'cpu')
        
    Returns:
        List of device IDs (integers for GPU, 'cpu' string for CPU)
    """
    if device_str.lower() == "cpu":
        return ["cpu"]
    return [int(d.strip()) for d in device_str.split(",")]