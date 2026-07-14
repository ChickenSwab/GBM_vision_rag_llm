# %%
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


#LAZY LOADING

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = None
model = None


def load_qwen():

    global tokenizer
    global model

    if model is None:

        print("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        print("Loading model...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map="auto"
        )

        model.eval()

        print("Qwen loaded successfully.")

    return tokenizer, model


import json

def build_prompt(rag_result):

    prompt = f"""
You are an expert neuro-oncology clinical decision support assistant.

You are NOT diagnosing the patient.

Your ONLY task is to summarize the evidence produced by:

• Swin UNETR MRI segmentation
• Radiomic feature extraction
• Retrieval-Augmented Generation (RAG) using the CGGA database

==================================================
IMPORTANT RULES
==================================================

Use ONLY the supplied information.

Never invent information.

Never assume molecular markers.

Never infer treatment response.

Never predict prognosis.

Never invent anatomy.

Do not infer disease severity beyond the supplied evidence.

If information is unavailable, explicitly state:

"Not available from the current evidence."

The retrieved cohort represents similar historical patients only.

The retrieved cohort does NOT represent the uploaded patient's molecular profile, prognosis, or treatment response.

==================================================
IMPORTANT TERMINOLOGY
==================================================

Use these definitions exactly.

WT = Whole Tumor

TC = Tumor Core

ET = Enhancing Tumor

NCR = Necrotic Core

ED = Peritumoral Edema

These are tumor regions.

They DO NOT mean:

- White Matter
- Gray Matter
- Cerebrospinal Fluid
- Normal Brain Tissue

Never redefine these abbreviations.

==================================================
VISION METRICS
==================================================

{json.dumps(rag_result["vision_metrics"], indent=2)}

==================================================
SEMANTIC FINDINGS
==================================================

{json.dumps(rag_result["semantic_findings"], indent=2)}

==================================================
CGGA COHORT SUMMARY
==================================================

Dominant IDH:
{rag_result["cohort_summary"]["dominant_idh"]}

Dominant MGMT:
{rag_result["cohort_summary"]["dominant_mgmt"]}

Average Overall Survival:
{rag_result["cohort_summary"]["average_os_days"]} days

Average Semantic Similarity:
{rag_result["cohort_summary"]["average_similarity_percent"]}%

Retrieved Patients:
{len(rag_result["retrieved_patients"])}

==================================================
TOP MATCH
==================================================

{json.dumps(rag_result["top_match"], indent=2)}

==================================================
TOP RETRIEVED PATIENTS
==================================================

{json.dumps(rag_result["retrieved_patients"], indent=2)}
The Top Match and Top Retrieved Patients are provided for context only.

Do NOT mention patient IDs.

Do NOT describe individual retrieved patients.

Use them ONLY to summarize the retrieved cohort statistics.
==================================================

Generate EXACTLY the following report.

# GBM Imaging Report

## 1. Tumor Characteristics

Report ONLY:

• Tumor location

• Whole Tumor (WT) volume

• Tumor Core (TC) volume

• Enhancing Tumor (ET) volume

• Necrotic Core (NCR) volume

• Edema volume

• Shape characteristics

• Enhancement ratio

Do NOT interpret beyond the supplied measurements.

--------------------------------------------------

## 2. Imaging Interpretation

Summarize ONLY the supplied semantic findings.

Do not introduce new imaging observations.

Do not mention:

- proliferation
- invasion
- necrosis
- aggressiveness

unless explicitly present in the semantic findings.

--------------------------------------------------

## 3. Similar CGGA Cohort

Summarize ONLY the retrieved CGGA cohort.

Include:

• Retrieved cohort size

• Average semantic similarity score

• Dominant IDH status

• Dominant MGMT status

• Average overall survival

State clearly that these findings summarize the retrieved cohort only.

Do NOT imply they represent the uploaded patient.

Do NOT mention individual patient IDs.

--------------------------------------------------

## 4. Molecular Evidence

Report ONLY the molecular characteristics of the retrieved cohort.

Include:

• Dominant IDH status

• Dominant MGMT status

• IDH distribution (if available)

• MGMT distribution (if available)

Always introduce this section using wording similar to:

"The retrieved cohort demonstrated the following molecular characteristics."

Never state or imply that the uploaded patient has these molecular characteristics.

Never predict mutation status.

Never reinterpret molecular terminology.

--------------------------------------------------

## 5. Treatment Considerations

ONLY mention treatments if they appear in the retrieved cohort.

If treatment information is unavailable, write:

"Treatment information was not available in the retrieved evidence."

Never recommend treatment.

--------------------------------------------------

## 6. Prognostic Discussion

Do NOT predict the prognosis of the uploaded patient.

Instead summarize ONLY the retrieved cohort survival statistics.

Example:

"The retrieved cohort demonstrated an average overall survival of X days."

Then add:

"This information summarizes similar retrieved cases and should not be interpreted as an individual patient prognosis."
--------------------------------------------------

## 7. Confidence Statement

This report was generated using:

• Swin UNETR MRI segmentation

• Quantitative radiomic feature extraction

• Retrieval-Augmented Generation (RAG) using the CGGA dataset

The report summarizes imaging-derived measurements together with retrieved cohort evidence.

The molecular findings describe the retrieved cohort only.

The retrieved cohort should not be interpreted as the molecular profile, prognosis, or treatment response of the uploaded patient.

This report is intended for research support and should be interpreted alongside expert clinical evaluation.

This report should not replace clinical judgement.
"""

    return prompt


# GENERATE REPORT USING QWEN
def generate_report(rag_result):

    tokenizer, model = load_qwen()

    # ---------------------------------------
    # Build Prompt
    # ---------------------------------------

    prompt = build_prompt(rag_result)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert neuro-oncology clinical "
                "decision support assistant."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Convert to Chat Format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)


    # Generate
    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=700,

            temperature=0.2,

            do_sample=True,

            top_p=0.9,

            repetition_penalty=1.1
        )


    # Decode
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    report = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return report

import gc
import torch


def unload_qwen():

    global tokenizer
    global model

    if model is not None:

        del model
        model = None

    if tokenizer is not None:

        del tokenizer
        tokenizer = None

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print("Qwen unloaded.")

