
# ============================================================
# COMPONENT 1 - SWIN UNETR INFERENCE
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import os
import numpy as np
from skimage.transform import resize
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
 
def extract_all_features(
    seg_mask,
    mri_tensor=None,
    affine=None
):
    """
    Extract vision metrics from the predicted BraTS segmentation.

    Parameters
    ----------
    seg_mask : numpy.ndarray
        3D segmentation mask with BraTS labels:
        0 = background
        1 = NCR
        2 = edema
        3 = enhancing tumor

    mri_tensor : numpy.ndarray, optional
        Shape: (4, H, W, D)
        Channels:
        0 = FLAIR
        1 = T1
        2 = T1CE
        3 = T2

    affine : numpy.ndarray, optional
        NIfTI affine matrix used to calculate physical voxel volume.

    Returns
    -------
    features : dict
        Extracted vision metrics.

    query : str
        Natural-language query for the RAG component.
    """

    features = {}

    # ============================================================
    # VOXEL VOLUME
    # ============================================================

    # Calculate physical voxel volume from the affine.
    # NIfTI affine is generally expressed in mm.
    # 1 cc = 1000 mm^3.

    if affine is not None:

        try:
            voxel_volume_mm3 = abs(
                np.linalg.det(
                    np.asarray(affine)[:3, :3]
                )
            )

            voxel_volume_cc = voxel_volume_mm3 / 1000.0

        except Exception:

            voxel_volume_cc = 0.001

    else:

        # Fallback for cases where affine is unavailable
        voxel_volume_cc = 0.001

    # ============================================================
    # BRATS LABEL DEFINITIONS
    # ============================================================

    # BraTS:
    # 1 = NCR / necrotic and non-enhancing tumor core
    # 2 = ED / edema
    # 3 = ET / enhancing tumor
    #
    # TC = NCR + ET
    # WT = NCR + ED + ET

    ncr_vox = int(np.sum(seg_mask == 1))
    ed_vox = int(np.sum(seg_mask == 2))
    et_vox = int(np.sum(seg_mask == 3))

    tc_vox = ncr_vox + et_vox
    wt_vox = ncr_vox + ed_vox + et_vox

    # ============================================================
    # PHYSICAL VOLUMES
    # ============================================================

    tc_vol = tc_vox * voxel_volume_cc
    wt_vol = wt_vox * voxel_volume_cc
    et_vol = et_vox * voxel_volume_cc
    ed_vol = ed_vox * voxel_volume_cc
    ncr_vol = ncr_vox * voxel_volume_cc

    features["wt_volume_cc"] = round(wt_vol, 2)
    features["tc_volume_cc"] = round(tc_vol, 2)
    features["et_volume_cc"] = round(et_vol, 2)
    features["ncr_volume_cc"] = round(ncr_vol, 2)
    features["ed_volume_cc"] = round(ed_vol, 2)

    # ============================================================
    # VOLUME RATIOS
    # ============================================================

    features["et_tc_ratio"] = (
        round(et_vol / tc_vol, 3)
        if tc_vol > 0
        else 0
    )

    features["ncr_tc_ratio"] = (
        round(ncr_vol / tc_vol, 3)
        if tc_vol > 0
        else 0
    )

    features["ed_wt_ratio"] = (
        round(ed_vol / wt_vol, 3)
        if wt_vol > 0
        else 0
    )

    # ============================================================
    # SPATIAL LOCATION
    # ============================================================

    if wt_vox > 0:

        coords = np.argwhere(seg_mask > 0)

        centroid = coords.mean(axis=0)

        H, W, D = seg_mask.shape

        if affine is not None:

            try:

                # Convert voxel centroid to world coordinates
                centroid_h = np.append(centroid, 1)

                world_centroid = (
                    np.asarray(affine) @ centroid_h
                )

                x, y, z = world_centroid[:3]

                # ------------------------------------------------
                # RAS coordinates
                # x < 0 = right
                # x > 0 = left
                # ------------------------------------------------

                features["hemisphere"] = (
                    "right"
                    if x < 0
                    else "left"
                )

                # ------------------------------------------------
                # Coronal location
                # ------------------------------------------------

                features["coronal_location"] = (
                    "frontal"
                    if y > 120
                    else "parieto-temporal"
                )

                # ------------------------------------------------
                # Axial location
                # ------------------------------------------------

                features["axial_location"] = (
                    "inferior"
                    if z < 80
                    else
                    "middle"
                    if z < 160
                    else
                    "superior"
                )

            except Exception:

                # Fallback to voxel coordinates
                features["hemisphere"] = (
                    "right"
                    if centroid[1] < W / 2
                    else "left"
                )

                features["axial_location"] = (
                    "inferior"
                    if centroid[0] < H / 3
                    else
                    "middle"
                    if centroid[0] < 2 * H / 3
                    else
                    "superior"
                )

                features["coronal_location"] = (
                    "frontal"
                    if centroid[2] < D / 3
                    else
                    "parieto-temporal"
                    if centroid[2] < 2 * D / 3
                    else
                    "occipital"
                )

        else:

            # ----------------------------------------------------
            # Fallback when affine is unavailable
            # ----------------------------------------------------

            features["hemisphere"] = (
                "right"
                if centroid[1] < W / 2
                else "left"
            )

            features["axial_location"] = (
                "inferior"
                if centroid[0] < H / 3
                else
                "middle"
                if centroid[0] < 2 * H / 3
                else
                "superior"
            )

            features["coronal_location"] = (
                "frontal"
                if centroid[2] < D / 3
                else
                "parieto-temporal"
                if centroid[2] < 2 * D / 3
                else
                "occipital"
            )

    else:

        features["hemisphere"] = "unknown"
        features["axial_location"] = "unknown"
        features["coronal_location"] = "unknown"

    # ============================================================
    # SHAPE FEATURES
    # ============================================================

    if wt_vox > 0:

        wt_mask = (
            seg_mask > 0
        ).astype(np.uint8)

        objects = ndimage.find_objects(wt_mask)

        if objects:

            bbox = objects[0]

            bh = bbox[0].stop - bbox[0].start
            bw = bbox[1].stop - bbox[1].start
            bd = bbox[2].stop - bbox[2].start

            # Physical bounding-box volume
            bbox_vol = (
                bh
                * bw
                * bd
                * voxel_volume_cc
            )

            features["solidity"] = (
                round(wt_vol / bbox_vol, 3)
                if bbox_vol > 0
                else 0
            )

            # Elongation
            dims = sorted(
                [bh, bw, bd]
            )

            features["elongation"] = (
                round(
                    dims[2] / dims[0],
                    3
                )
                if dims[0] > 0
                else 0
            )

            # Sphericity
            grad = np.gradient(
                wt_mask.astype(float)
            )

            surf_vox = np.sum(
                np.sqrt(
                    sum(
                        g ** 2
                        for g in grad
                    )
                ) > 0.5
            )

            features["sphericity"] = (
                round(
                    (
                        np.pi ** (1 / 3)
                        * (6 * wt_vox) ** (2 / 3)
                    )
                    / surf_vox,
                    3
                )
                if surf_vox > 0
                else 0
            )

        else:

            features["solidity"] = 0
            features["elongation"] = 0
            features["sphericity"] = 0

    else:

        features["solidity"] = 0
        features["elongation"] = 0
        features["sphericity"] = 0

    # ============================================================
    # INTENSITY FEATURES
    # ============================================================

    if mri_tensor is not None:

        tc_mask = (
            (seg_mask == 1)
            | (seg_mask == 3)
        )

        modality_names = [
            "FLAIR",
            "T1",
            "T1CE",
            "T2"
        ]

        regions = [
            ("wt", seg_mask > 0),
            ("tc", tc_mask),
            ("et", seg_mask == 3)
        ]

        for i, name in enumerate(
            modality_names
        ):

            vol = mri_tensor[i]

            for region, cond in regions:

                if np.sum(cond) < 10:
                    continue

                vox = vol[cond]

                features[
                    f"{name}_{region}_mean"
                ] = round(
                    float(np.mean(vox)),
                    4
                )

                features[
                    f"{name}_{region}_std"
                ] = round(
                    float(np.std(vox)),
                    4
                )

        # ========================================================
        # ENHANCEMENT RATIO
        # ========================================================

        normal_mask = (
            seg_mask == 0
        )

        et_mask = (
            seg_mask == 3
        )

        if (
            np.sum(normal_mask) > 100
            and np.sum(et_mask) > 10
        ):

            normal_t1ce = np.clip(
                mri_tensor[2][normal_mask],
                0,
                None
            )

            et_t1ce = np.clip(
                mri_tensor[2][et_mask],
                0,
                None
            )

            mean_normal = float(
                np.mean(normal_t1ce)
            )

            mean_et = float(
                np.mean(et_t1ce)
            )

            features["enhancement_ratio"] = round(
                (
                    mean_et / mean_normal
                    if mean_normal > 0.01
                    else 0
                ),
                3
            )

        else:

            features["enhancement_ratio"] = 0

    else:

        features["enhancement_ratio"] = 0

    # ============================================================
    # RAG QUERY
    # ============================================================

    query = (
        f"Glioma in "
        f"{features['hemisphere']} hemisphere, "
        f"{features['coronal_location']} "
        f"{features['axial_location']} region. "

        f"Whole tumor "
        f"{features['wt_volume_cc']}cc, "

        f"tumor core "
        f"{features['tc_volume_cc']}cc "

        f"(NCR "
        f"{features['ncr_volume_cc']}cc + "

        f"ET "
        f"{features['et_volume_cc']}cc), "

        f"edema "
        f"{features['ed_volume_cc']}cc. "

        f"ET/TC ratio "
        f"{features['et_tc_ratio']}, "

        f"NCR/TC ratio "
        f"{features['ncr_tc_ratio']}, "

        f"edema/WT ratio "
        f"{features['ed_wt_ratio']}, "

        f"solidity "
        f"{features.get('solidity', 'N/A')}, "

        f"elongation "
        f"{features.get('elongation', 'N/A')}, "

        f"sphericity "
        f"{features.get('sphericity', 'N/A')}. "

        f"Enhancement ratio "
        f"{features.get('enhancement_ratio', 'N/A')}. "

        f"What is the WHO grade, IDH status, "
        f"MGMT methylation, treatment protocol "
        f"and prognosis?"
    )

    return features, query

#visualization
def plot_segmentation_overlay(
    mri_numpy,
    seg_mask,
    case_id,
    raw_flair_path=None,
):
    """
    Saves an overlay of the predicted segmentation.

    Returns
    -------
    str
        Path to saved PNG.
    """

    if raw_flair_path is not None and os.path.exists(raw_flair_path):
        raw = nib.load(raw_flair_path).get_fdata()

        if raw.shape == seg_mask.shape:
            flair = raw
        else:
            flair = mri_numpy[0]
    else:
        flair = mri_numpy[0]

    p1, p99 = np.percentile(flair[flair > 0], [1, 99])

    flair = np.clip(flair, p1, p99)
    flair = (flair - p1) / (p99 - p1 + 1e-8)

    tumour = (seg_mask > 0).astype(int)

    # -------------------------------------------------
    # SAVE RAW MRI
    # -------------------------------------------------

    fig_raw, axes_raw = plt.subplots(1, 3, figsize=(16,5))
    fig_raw.suptitle(f"{case_id} - Raw MRI", fontsize=13)

    names = [
        "Axial",
        "Coronal",
        "Sagittal"
    ]

    for ax, axis, title in zip(axes_raw, [0,1,2], names):

        if axis == 0:

            counts = tumour.sum(axis=(1,2))
            sl = np.argmax(counts)

            fl = flair[sl,:,:]

        elif axis == 1:

            counts = tumour.sum(axis=(0,2))
            sl = np.argmax(counts)

            fl = flair[:,sl,:]

        else:

            counts = tumour.sum(axis=(0,1))
            sl = np.argmax(counts)

            fl = flair[:,:,sl]

        ax.imshow(
            fl.T,
            cmap="gray",
            origin="lower"
        )

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)

    raw_path = os.path.join(
        "outputs",
        f"{case_id}_raw.png"
    )

    plt.savefig(
        raw_path,
        dpi=150
    )

    plt.close(fig_raw)

    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    fig.suptitle(f"{case_id}", fontsize=13)

    names = [
        "Axial",
        "Coronal",
        "Sagittal"
    ]

    for ax, axis, title in zip(axes,[0,1,2],names):

        if axis == 0:

            counts = tumour.sum(axis=(1,2))
            sl = np.argmax(counts)

            fl = flair[sl,:,:]
            mk = seg_mask[sl,:,:]

        elif axis == 1:

            counts = tumour.sum(axis=(0,2))
            sl = np.argmax(counts)

            fl = flair[:,sl,:]
            mk = seg_mask[:,sl,:]

        else:

            counts = tumour.sum(axis=(0,1))
            sl = np.argmax(counts)

            fl = flair[:,:,sl]
            mk = seg_mask[:,:,sl]

        if mk.shape != fl.shape:

            mk = resize(
                mk,
                fl.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.int32)

        overlay = np.zeros((*fl.shape,4))

        overlay[mk==1] = [1,0,0,0.6]
        overlay[mk==2] = [1,1,0,0.4]
        overlay[mk==3] = [0,0.3,1,0.7]

        ax.imshow(fl.T,cmap="gray",origin="lower")
        ax.imshow(overlay.transpose(1,0,2),origin="lower")

        ax.set_title(title)
        ax.axis("off")

    legend = [

        mpatches.Patch(color="red",label="NCR"),

        mpatches.Patch(color="yellow",label="Edema"),

        mpatches.Patch(color="blue",label="Enhancing Tumor"),

    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3
    )

    plt.tight_layout(rect=[0,0.05,1,1])

    os.makedirs("outputs",exist_ok=True)

    save_path = os.path.join(
        "outputs",
        f"{case_id}_overlay.png"
    )

    plt.savefig(save_path,dpi=150)

    plt.close()

    return raw_path, save_path

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

    print("1. Dataset created")

    sample = dataset[0]

    print("2. Sample loaded")

    image = (
        sample["image"]
        .unsqueeze(0)
        .to(device)
        .float()
    )

    print("3. Image shape:", image.shape)

    if torch.cuda.is_available():
        print(
            f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB"
        )
        print(
            f"Reserved : {torch.cuda.memory_reserved()/1024**2:.1f} MB"
        )

    print("4. Starting inference...")

    model.eval()

    with torch.no_grad():

        prediction = sliding_window_inference(
            inputs=image,
            roi_size=(96, 96, 96),   # or 96 if you reverted it
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
            mode="gaussian"
        )

    print("5. Inference finished")


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

    raw_path, overlay_path = plot_segmentation_overlay(
        mri_numpy=mri_numpy,
        seg_mask=seg_mask,
        case_id="Uploaded_Case",
        raw_flair_path=flair_path,
    )
    # RETURN EVERYTHING
    return {

        "raw_image": raw_path,

        "overlay": overlay_path,

        "segmentation": seg_mask,

        "features": features,

        "query": query

    }

