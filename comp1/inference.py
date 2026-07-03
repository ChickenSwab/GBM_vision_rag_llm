
# ============================================================
# COMPONENT 1 - SWIN UNETR INFERENCE
# ============================================================

import os
import numpy as np
import torch

from scipy import ndimage

from monai.networks.nets import SwinUNETR

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConcatItemsd,
    DeleteItemsd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    ToTensord,
)

from monai.data import Dataset

from monai.inferers import sliding_window_inference


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# MRI MODALITIES
# ============================================================

MODALITY_KEYS = [
    "flair",
    "t1",
    "t1ce",
    "t2",
]


# ============================================================
# LOAD SWIN UNETR MODEL
# ============================================================

def load_model(model_path):

    model = SwinUNETR(
        in_channels=4,
        out_channels=4,
        feature_size=48,
        spatial_dims=3,
        dropout_path_rate=0.1,
    ).to(device)

    print("Loading checkpoint...")

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if "model_state" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD MODEL
# ============================================================

# MODEL_PATH = "best_model.pth"

# model = load_model(MODEL_PATH)

# %%
# ============================================================
# TRANSFORMS FOR UPLOADED MRI
# ============================================================

upload_transforms = Compose(

    [

        # ----------------------------------------------------
        # LOAD MRI MODALITIES
        # ----------------------------------------------------

        LoadImaged(
            keys=MODALITY_KEYS
        ),

        # ----------------------------------------------------
        # ENSURE CHANNEL FIRST
        # ----------------------------------------------------

        EnsureChannelFirstd(
            keys=MODALITY_KEYS
        ),

        # ----------------------------------------------------
        # COMBINE INTO 4-CHANNEL IMAGE
        # ----------------------------------------------------

        ConcatItemsd(

            keys=MODALITY_KEYS,

            name="image",

            dim=0

        ),

        # ----------------------------------------------------
        # REMOVE INDIVIDUAL MODALITIES
        # ----------------------------------------------------

        DeleteItemsd(
            keys=MODALITY_KEYS
        ),

        # ----------------------------------------------------
        # ORIENTATION
        # ----------------------------------------------------

        Orientationd(

            keys=["image"],

            axcodes="RAS"

        ),

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        NormalizeIntensityd(

            keys="image",

            nonzero=True,

            channel_wise=True

        ),

        # ----------------------------------------------------
        # REMOVE EMPTY BACKGROUND
        # ----------------------------------------------------

        CropForegroundd(

            keys=["image"],

            source_key="image"

        ),

        # ----------------------------------------------------
        # PAD TO MODEL INPUT SIZE
        # ----------------------------------------------------

        SpatialPadd(

            keys=["image"],

            spatial_size=(128, 128, 128)

        ),

        # ----------------------------------------------------
        # CONVERT TO TENSOR
        # ----------------------------------------------------

        ToTensord(

            keys=["image"]

        )

    ]

)

# %%
from scipy import ndimage
 
def extract_all_features(seg_mask, mri_tensor=None, affine=None, voxel_volume_cc=0.001):
    """
    seg_mask   : (H, W, D) numpy array, labels 0/1/2/3
    mri_tensor : (4, H, W, D) numpy array — FLAIR, T1, T1CE, T2
    returns    : (features dict, clinical query string)
    """
    features = {}
 
    # BraTS label definitions
    # 1 = NCR (necrotic core)
    # 2 = ED  (edema)
    # 3 = ET  (enhancing tumor)
    # TC = NCR + ET
    # WT = NCR + ED + ET
 
    ncr_vox = int(np.sum(seg_mask == 1))
    ed_vox  = int(np.sum(seg_mask == 2))
    et_vox  = int(np.sum(seg_mask == 3))
    tc_vox  = ncr_vox + et_vox
    wt_vox  = ncr_vox + ed_vox + et_vox
 
    tc_vol  = tc_vox  * voxel_volume_cc
    wt_vol  = wt_vox  * voxel_volume_cc
    et_vol  = et_vox  * voxel_volume_cc
    ed_vol  = ed_vox  * voxel_volume_cc
    ncr_vol = ncr_vox * voxel_volume_cc
 
    features["wt_volume_cc"]  = round(wt_vol,  2)
    features["tc_volume_cc"]  = round(tc_vol,  2)
    features["et_volume_cc"]  = round(et_vol,  2)
    features["ncr_volume_cc"] = round(ncr_vol, 2)
    features["ed_volume_cc"]  = round(ed_vol,  2)
    features["et_tc_ratio"]   = round(et_vol  / tc_vol,  3) if tc_vol  > 0 else 0
    features["ncr_tc_ratio"]  = round(ncr_vol / tc_vol,  3) if tc_vol  > 0 else 0
    features["ed_wt_ratio"]   = round(ed_vol  / wt_vol,  3) if wt_vol  > 0 else 0
 
    # spatial location — RAS space: low W = right, high W = left
    # spatial location using RAS world coordinates
    if wt_vox > 0:

        coords = np.argwhere(seg_mask > 0)
        centroid = coords.mean(axis=0)

        H, W, D = seg_mask.shape

        if affine is not None:

        # convert voxel centroid -> world RAS
            centroid_h = np.append(centroid, 1)
            world_centroid = affine @ centroid_h

            x, y, z = world_centroid[:3]

        # RAS:
        # x < 0 = right hemisphere
        # x > 0 = left hemisphere
            features["hemisphere"] = ("right" if x < 0 else "left")

        # y axis: anterior/posterior
            features["coronal_location"] = (
                "frontal" if y > 120 else
                "parieto-temporal"
            )

        # z axis: inferior/superior
            features["axial_location"] = (
                "inferior" if z < 80 else
                "middle" if z < 160 else
                "superior"
            )

        else:

        # fallback if affine unavailable
            features["hemisphere"] = (
                "right" if centroid[1] < W/2 else "left"
            )

            features["axial_location"] = (
                "inferior" if centroid[0] < H/3 else
                "middle" if centroid[0] < 2*H/3 else
                "superior"
            )

            features["coronal_location"] = (
                "frontal" if centroid[2] < D/3 else
                "parieto-temporal" if centroid[2] < 2*D/3 else
                "occipital"
            )


    else:

        features["hemisphere"] = "unknown"
        features["axial_location"] = "unknown"
        features["coronal_location"] = "unknown"
    
 
    # shape features (Zwanenburg et al. 2020)
    if wt_vox > 0:
        wt_mask  = (seg_mask > 0).astype(np.uint8)
        bbox     = ndimage.find_objects(wt_mask)[0]
        bh = bbox[0].stop - bbox[0].start
        bw = bbox[1].stop - bbox[1].start
        bd = bbox[2].stop - bbox[2].start
        bbox_vol = bh * bw * bd * voxel_volume_cc
 
        features["solidity"]   = round(wt_vol / bbox_vol, 3) if bbox_vol > 0 else 0
        dims = sorted([bh, bw, bd])
        features["elongation"] = round(dims[2] / dims[0], 3) if dims[0] > 0 else 0
 
        grad     = np.gradient(wt_mask.astype(float))
        surf_vox = np.sum(np.sqrt(sum(g**2 for g in grad)) > 0.5)
        features["sphericity"] = round(
            (np.pi**(1/3) * (6 * wt_vox)**(2/3)) / surf_vox, 3
        ) if surf_vox > 0 else 0
 
    # intensity features
    if mri_tensor is not None:
        tc_mask = (seg_mask == 1) | (seg_mask == 3)
 
        for i, name in enumerate(["FLAIR", "T1", "T1CE", "T2"]):
            vol = mri_tensor[i]
            for region, cond in [
                ("wt", seg_mask > 0),
                ("tc", tc_mask),
                ("et", seg_mask == 3),
            ]:
                if np.sum(cond) < 10:
                    continue
                vox = vol[cond]
                features[f"{name}_{region}_mean"] = round(float(np.mean(vox)), 4)
                features[f"{name}_{region}_std"]  = round(float(np.std(vox)),  4)
 
        # enhancement ratio (Ellingson et al. 2017)
        normal_mask = seg_mask == 0
        et_mask     = seg_mask == 3
        if np.sum(normal_mask) > 100 and np.sum(et_mask) > 10:
            mean_normal = float(np.mean(np.clip(mri_tensor[2][normal_mask], 0, None)))
            mean_et     = float(np.mean(np.clip(mri_tensor[2][et_mask],     0, None)))
            features["enhancement_ratio"] = round(
                mean_et / mean_normal if mean_normal > 0.01 else 0, 3
            )
 
    query = (
        f"Glioma in {features['hemisphere']} hemisphere, "
        f"{features['coronal_location']} {features['axial_location']} region. "
        f"Whole tumor {features['wt_volume_cc']}cc, "
        f"tumor core {features['tc_volume_cc']}cc "
        f"(NCR {features['ncr_volume_cc']}cc + ET {features['et_volume_cc']}cc), "
        f"edema {features['ed_volume_cc']}cc. "
        f"ET/TC ratio {features['et_tc_ratio']}, "
        f"NCR/TC ratio {features['ncr_tc_ratio']}, "
        f"edema/WT ratio {features['ed_wt_ratio']}, "
        f"solidity {features.get('solidity', 'N/A')}, "
        f"sphericity {features.get('sphericity', 'N/A')}. "
        f"Enhancement ratio {features.get('enhancement_ratio', 'N/A')}. "
        f"What is the WHO grade, IDH status, MGMT methylation, "
        f"treatment protocol and prognosis?"
    )
 
    return features, query

# run inference on an uploaded mri

def run_uploaded_case(
    flair_path,
    t1_path,
    t1ce_path,
    t2_path,
    model
):
    """
    Performs inference on an uploaded MRI.

    Parameters
    ----------
    flair_path : str
    t1_path    : str
    t1ce_path  : str
    t2_path    : str
    model      : Loaded Swin UNETR model

    Returns
    -------
    dict containing:
        segmentation
        features
        query
    """


    patient = {

        "flair": flair_path,

        "t1": t1_path,

        "t1ce": t1ce_path,

        "t2": t2_path

    }

    dataset = Dataset(
        data=[patient],
        transform=upload_transforms
    )

    sample = dataset[0]

    image = (
        sample["image"]
        .unsqueeze(0)
        .to(device)
        .float()
    )

    # RUN SWIN UNETR    
    model.eval()

    with torch.no_grad():

        prediction = sliding_window_inference(

            inputs=image,

            roi_size=(96, 96, 96),

            sw_batch_size=2,

            predictor=model,

            overlap=0.5,

            mode="gaussian"

        )


    # SEGMENTATION MASK
    seg_mask = (
        torch.argmax(
            prediction,
            dim=1
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    mri_numpy = (
        image
        .squeeze(0)
        .cpu()
        .numpy()
    )


    affine = None

    affine = None

    try:
        affine = sample["image_meta_dict"]["affine"]
    except Exception:
        affine = None

    # FEATURE EXTRACTION
    features, query = extract_all_features(

        seg_mask,

        mri_numpy,

        affine

    )

    # RETURN EVERYTHING
    return {

        "segmentation": seg_mask,

        "features": features,

        "query": query

    }

