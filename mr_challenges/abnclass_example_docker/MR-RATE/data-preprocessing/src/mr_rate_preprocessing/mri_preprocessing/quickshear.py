"""
MR-RATE MRI Preprocessing Pipeline - Quickshear Defacing


Quickshear and BrainLesion Suite Attribution
--------------------------------------------
apply_defacing function in this script is implemented based on the Defacer.apply_mask function from the BrainLesion Suite preprocessing repository:
https://github.com/BrainLesion/preprocessing/blob/main/brainles_preprocessing/defacing/defacer.py

generate_defacing_mask function in this script is implemented based on 
    the QuickshearDefacer.deface function and the nipy_quickshear.py script from the preprocessing repository:
    https://github.com/BrainLesion/preprocessing/blob/main/brainles_preprocessing/defacing/quickshear/quickshear.py
    https://github.com/BrainLesion/preprocessing/blob/main/brainles_preprocessing/defacing/quickshear/nipy_quickshear.py
and 
    the quickshear.py script from the quickshear repository:
    https://github.com/nipy/quickshear/blob/master/quickshear.py

If you are using this script that uses Quickshear and BrainLesion Suite, please cite the following publications:

Schimke, Nakeisha, and John Hale. "Quickshear defacing for neuroimages." Proceedings 
of the 2nd USENIX conference on Health security and privacy. USENIX Association, 2011.

Kofler, F., Rosier, M., Astaraki, M., Möller, H., Mekki, I. I., Buchner, J. A., Schmick, 
A., Pfiffer, A., Oswald, E., Zimmer, L., Rosa, E. de la, Pati, S., Canisius, J., Piffer, 
A., Baid, U., Valizadeh, M., Linardos, A., Peeken, J. C., Shit, S., … Menze, B. (2025). 
BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image 
Analysis https://arxiv.org/abs/2507.09036
"""

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve
import nibabel as nib

from utils import BufferedStudyLogger
from brainles_preprocessing.defacing.quickshear.nipy_quickshear import convex_hull
from auxiliary.io import read_image, write_image


def generate_defacing_mask(
    brain_mask_path: Path,
    defacing_mask_path: Path,
    logger: Optional[BufferedStudyLogger] = None,
) -> bool:
    """
    This function is adapted from
        the QuickshearDefacer.deface function and the nipy_quickshear.py script from the preprocessing repository:
        https://github.com/BrainLesion/preprocessing/blob/main/brainles_preprocessing/defacing/quickshear/quickshear.py
        https://github.com/BrainLesion/preprocessing/blob/main/brainles_preprocessing/defacing/quickshear/nipy_quickshear.py
    and 
        the quickshear.py script from the quickshear repository:
        https://github.com/nipy/quickshear/blob/master/quickshear.py
    
    Modifications are done to be used as a standalone function in this pipeline and fix 2 issues:
        - When brain touches the edges of the image, edge_mask function fails because np.roll wraps around the edge pixels
        - When the brain P and S axes of image are not in good ratio, the defacing mask is not generated correctly
        See https://github.com/BrainLesion/preprocessing/issues/167 for more details


    Generate a defacing mask from a brain mask using Quickshear and save it as np.uint8 NIfTI file.
    
    Args:
        brain_mask_path: Path to brain mask
        defacing_mask_path: Path to save defacing mask
        logger: Optional logger instance
        
    Returns:
        True if successful, False otherwise
    """
    def edge_mask(mask):
        """Find the edges of a mask or masked image

        Parameters
        ----------
        mask : 3D array
            Binary mask (or masked image) with axis orientation LPS or RPS, and the
            non-brain region set to 0

        Returns
        -------
        2D array
            Outline of sagittal profile (PS orientation) of mask
        """
        # Sagittal profile
        brain = mask.any(axis=0)

        # Simple edge detection
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        edgemask = convolve(brain, kernel, mode="constant", cval=0.0) != 0 # Was using np.roll originally !!!

        return edgemask.astype("uint8")

    def run_quickshear(bet_img: nib.nifti1.Nifti1Image, buffer: int = 10) -> NDArray:
        """Deface image using Quickshear algorithm

        Parameters
        ----------
        bet_img : Nifti1Image
            Nibabel image of skull-stripped brain mask or masked anatomical
        buffer : int
            Distance from mask to set shearing plane

        Returns
        -------
        defaced_mask: NDArray
            Defaced image mask
        """
        src_ornt = nib.io_orientation(bet_img.affine)
        tgt_ornt = nib.orientations.axcodes2ornt("RPS")
        to_RPS = nib.orientations.ornt_transform(src_ornt, tgt_ornt)
        from_RPS = nib.orientations.ornt_transform(tgt_ornt, src_ornt)

        mask_RPS = nib.orientations.apply_orientation(bet_img.dataobj, to_RPS)

        edgemask = edge_mask(mask_RPS)
        low = convex_hull(edgemask)
        xdiffs, ydiffs = np.diff(low)
        slope = ydiffs[0] / xdiffs[0]

        yint = low[1][0] - (low[0][0] * slope) - buffer
        ys = np.arange(0, mask_RPS.shape[1]) * slope + yint # Was using shape[2] originally !!!
        defaced_mask_RPS = np.ones(mask_RPS.shape, dtype="bool")

        for x, y in zip(np.nonzero(ys > 0)[0], ys.astype(int)):
            defaced_mask_RPS[:, x, :y] = 0

        defaced_mask = nib.orientations.apply_orientation(defaced_mask_RPS, from_RPS)

        return defaced_mask

    try:
        # Load the brain mask
        brain_mask = nib.load(str(brain_mask_path))

        # Generate the defacing mask using Quickshear
        defacing_mask = run_quickshear(bet_img=brain_mask, buffer=10.0).astype("uint8")

        # Transpose to match SimpleITK order needed by write_image
        defacing_mask = defacing_mask.transpose(2, 1, 0)
        
        # Save the defacing mask
        write_image(
            input_array=defacing_mask,
            output_path=str(defacing_mask_path),
            reference_path=str(brain_mask_path),
            create_parent_directory=True,
        )
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Defacing mask generation failed: {e}")
        return False


def apply_defacing(
    input_path: Path,
    defacing_mask_path: Path,
    output_path: Path,
    logger: Optional[BufferedStudyLogger] = None,
) -> bool:
    """
    This function (originally self.apply_mask) is adapted from https://github.com/BrainLesion/preprocessing/blob/main/brainles_preprocessing/defacing/defacer.py 
    Slight modifications, mainly to control output dtype, are done to be used as a standalone function in this pipeline.


    Apply a defacing mask to an image. The output dtype of the defaced image is determined 
    automatically: if masked data is losslessly representable within uint16 range, the image 
    is saved as uint16; otherwise float32 is used.
    
    Args:
        input_path: Path to input image
        defacing_mask_path: Path to defacing mask
        output_path: Path to save defaced image
        logger: Optional logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read input and mask data
        input_data = read_image(str(input_path))
        mask_data = read_image(str(defacing_mask_path))
        
        # Apply mask
        masked_data = np.where(mask_data.astype(bool), input_data, 0)

        # Determine if uint16 is a suitable output dtype, use float32 otherwise
        if (masked_data.max() <= np.iinfo(np.uint16).max
                and np.array_equal(masked_data, np.floor(masked_data))
                and masked_data.min() >= np.iinfo(np.uint16).min):
            out_dtype = np.uint16
        else:
            out_dtype = np.float32
        masked_data = masked_data.astype(out_dtype)

        # Save the defaced image
        write_image(
            input_array=masked_data,
            output_path=str(output_path),
            reference_path=str(input_path),
            create_parent_directory=True,
        )
                
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Defacing application failed: {e}")
        return False