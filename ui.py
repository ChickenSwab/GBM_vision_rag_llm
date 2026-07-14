from pathlib import Path
import traceback
from nicegui import ui
from app import analyze_patient

# ---------------------------------------------------------
# Upload folder
# ---------------------------------------------------------

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

uploaded = {
    "flair": None,
    "t1": None,
    "t1ce": None,
    "t2": None,
}


# ---------------------------------------------------------
# SAVE UPLOADED FILE
# ---------------------------------------------------------

async def save_file(e, key):

    path = UPLOAD_DIR / e.file.name

    await e.file.save(path)

    uploaded[key] = str(path)

    ui.notify(f"{key.upper()} uploaded successfully.")


# ---------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------

async def analyze():

    if None in uploaded.values():

        ui.notify(
            "Please upload all four MRI files.",
            color="negative",
        )

        return

    status.text = "Running pipeline..."

    try:

        result = analyze_patient(

            uploaded["flair"],

            uploaded["t1"],

            uploaded["t1ce"],

            uploaded["t2"]

        )

        status.text = "Analysis Complete"

        # -------------------------
        # Display segmentation
        # -------------------------

        raw_image.set_source(
            result["raw_image"]
        )

        overlay.set_source(
            result["overlay"]
        )
        # -------------------------
        # Display report
        # -------------------------
        
        report_box.set_content(
            result["report"]
        )

        ui.notify(
            "Analysis completed.",
            color="positive",
        )

    except Exception as ex:

        traceback.print_exc()

        status.text = "Pipeline Failed"

        report_box.set_content(
            f"""```text
    {traceback.format_exc()}
    ```"""
        )


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

ui.label(
    "GBM Multimodal Analysis System"
).classes(
    "text-3xl font-bold"
)

ui.separator()

ui.upload(
    label="FLAIR MRI",
    auto_upload=True,
    on_upload=lambda e: save_file(e, "flair"),
)

ui.upload(
    label="T1 MRI",
    auto_upload=True,
    on_upload=lambda e: save_file(e, "t1"),
)

ui.upload(
    label="T1CE MRI",
    auto_upload=True,
    on_upload=lambda e: save_file(e, "t1ce"),
)

ui.upload(
    label="T2 MRI",
    auto_upload=True,
    on_upload=lambda e: save_file(e, "t2"),
)

ui.separator()

ui.button(
    "Analyze",
    on_click=analyze,
).classes("w-40")

status = ui.label("Ready")

ui.separator()

with ui.row():

    with ui.column():

        ui.label(
            "Original MRI"
        ).classes(
            "text-xl font-bold"
        )

        raw_image = ui.image().classes(
            "w-[550px] rounded-lg shadow-lg"
        )

    with ui.column():

        ui.label(
            "Segmentation Overlay"
        ).classes(
            "text-xl font-bold"
        )

        overlay = ui.image().classes(
            "w-[550px] rounded-lg shadow-lg"
        )

ui.separator()

ui.label(
    "Clinical Report"
).classes(
    "text-xl font-bold"
)

report_box = ui.markdown()

ui.run(
    title="GBM Multimodal Analysis System"
)