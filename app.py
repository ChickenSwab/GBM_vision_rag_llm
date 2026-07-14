import os
import gc
import torch

from comp1.inference import (
    load_model,
    run_uploaded_case
)

from comp2.rag import (
    run_rag,
    unload_encoder
)

from comp3.llm import (
    generate_report,
    unload_qwen
)


MODEL_PATH = os.path.join(
    "models",
    "best_model.pth"
)
def get_vision_model():

    global vision_model

    if vision_model is None:

        print("Loading Swin UNETR...")

        vision_model = load_model(
            MODEL_PATH
        )

        print("Vision model loaded.\n")

    return vision_model
vision_model = None
# print("Loading Swin UNETR...")

# vision_model = load_model(MODEL_PATH)

# print("Vision model loaded.\n")


# UNLOAD VISION MODEL
def unload_vision_model():

    global vision_model

    if vision_model is not None:

        del vision_model
        vision_model = None

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print("Vision model unloaded.\n")


# ANALYZE A SINGLE MRI CASE
def analyze_patient(
    flair_path,
    t1_path,
    t1ce_path,
    t2_path
):

    print("\n===================================")
    print("STEP 1 : Running Vision Module")
    print("===================================\n")

    vision_result = run_uploaded_case(

        flair_path=flair_path,

        t1_path=t1_path,

        t1ce_path=t1ce_path,

        t2_path=t2_path,

        model=get_vision_model()

    )

    print("Vision analysis completed.\n")

    # print(
    #     f"Segmentation overlay saved to:\n"
    #     f"{vision_result['overlay']}\n"
    # )

    # FREE SWIN MODEL
    # unload_vision_model()

    print("===================================")
    print("STEP 2 : Running RAG Retrieval")
    print("===================================\n")

    rag_result = run_rag(

        vision_result["features"],

        vision_result["query"]

    )

    print("RAG retrieval completed.\n")


    # FREE BGE
    unload_encoder()

    print("===================================")
    print("STEP 3 : Generating Clinical Report")
    print("===================================\n")

    report = generate_report(
        rag_result
    )

    print("Report generation completed.\n")

    # FREE QWEN
    unload_qwen()

    return {

        "report": report,

        "raw_image": vision_result["raw_image"],

        "overlay": vision_result["overlay"],

        "features": vision_result["features"],

        "rag": rag_result

    }


# MAIN
if __name__ == "__main__":

    print("\n==========================================")
    print(" GBM MULTIMODAL ANALYSIS SYSTEM")
    print("==========================================\n")

    flair_path = input(
        "Enter FLAIR MRI path : "
    ).strip()

    t1_path = input(
        "Enter T1 MRI path    : "
    ).strip()

    t1ce_path = input(
        "Enter T1CE MRI path  : "
    ).strip()

    t2_path = input(
        "Enter T2 MRI path    : "
    ).strip()

    print("\nStarting analysis...\n")

    report = analyze_patient(

        flair_path,

        t1_path,

        t1ce_path,

        t2_path

    )

    print("\n")
    print("=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(report["report"])
    print("=" * 80)

import atexit

atexit.register(unload_vision_model)