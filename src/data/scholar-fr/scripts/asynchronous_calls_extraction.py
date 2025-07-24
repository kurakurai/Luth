import os
import fitz
import json
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import pandas as pd
import asyncio
import re
import hashlib
import time
import datetime
import uuid
from prompts import Prompts

#setup
PROJECT_ID = "" #Put project ID
LOCATION = "" #Put zone
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-flash-lite")

# --- JSON Utilities ---
def try_parse_json(text):
    try:
        return json.loads(text), "json"
    except json.JSONDecodeError:
        return text.strip(), "raw"

# --- Prompt call ---
def call_gemini(prompt):
    config = GenerationConfig(response_mime_type="application/json")
    try:
        response = model.generate_content(
            prompt,
            generation_config=config)
        return response.text.strip()
    except Exception as e:
        return f"ERREUR: {e}"

async def async_call_gemini(prompt, loop):
    return await loop.run_in_executor(None, call_gemini, prompt)

# --- PDF ---
def extract_text(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

# --- Étapes du pipeline ---

async def extract_questions(subject_text, loop):
    prompt = Prompts.extract_questions(subject_text)
    result = await async_call_gemini(prompt, loop)
    data, mode = try_parse_json(result)
    return data

async def filter_questions(questions, loop):
    if isinstance(questions, list):
        joined = "\n".join(questions)
    else:
        joined = questions
    prompt = Prompts.filter_questions(joined)
    result = await async_call_gemini(prompt, loop)
    return [q.strip() for q in result.split("\n") if q.strip()]

async def add_context(subject_text, questions, loop):
    prompt = Prompts.add_context_to_questions(subject_text, questions)
    result = await async_call_gemini(prompt, loop)
    data, mode = try_parse_json(result)
    return data

async def extract_answers(contexted_questions, correction_text, loop):
    prompt = Prompts.extact_answers(contexted_questions, correction_text)
    result = await async_call_gemini(prompt, loop)
    data, mode = try_parse_json(result)
    return data

async def enrich_context(answers, loop):
    prompt = Prompts.enrich_context(answers)
    result = await async_call_gemini(prompt, loop)
    data, mode = try_parse_json(result)
    return data

async def full_pipeline_async(subject_path, correction_path, loop, progress_callback=None):
    try:
        if progress_callback:
            await progress_callback("Extraction du texte")
        subject_text = await loop.run_in_executor(None, extract_text, subject_path)
        correction_text = await loop.run_in_executor(None, extract_text, correction_path)

        if progress_callback:
            await progress_callback("Détection des questions")
        questions = await extract_questions(subject_text, loop)

        if progress_callback:
            await progress_callback("Ajout du contexte")
        contexted = await add_context(subject_text, questions, loop)

        if progress_callback:
            await progress_callback("Extraction des réponses")
        answers = await extract_answers(contexted, correction_text, loop)

        if progress_callback:
            await progress_callback("Enrichissement du contexte")
        enriched = await enrich_context(answers, loop)

        return enriched
    except Exception as e:
        print(f"Error in pipeline: {subject_path}: {e}")
        return []

# --- Fichiers ---
def get_safe_filename(subject_path):
    base_name = os.path.splitext(os.path.basename(subject_path))[0]
    base_name = re.sub(r'[<>:"/\\|?*\s]', '_', base_name)
    base_name = re.sub(r'_{2,}', '_', base_name)
    if len(base_name) > 200 or not base_name.strip('_'):
        hash_name = hashlib.md5(subject_path.encode()).hexdigest()[:16]
        base_name = f"file_{hash_name}"
    return base_name

def file_already_processed(subject_path, output_dir):
    if not os.path.exists(output_dir):
        return False
    safe_name = get_safe_filename(subject_path)
    expected_output = os.path.join(output_dir, f"{safe_name}.json")
    return os.path.exists(expected_output)

def save_json(data, subject_path, output_dir):
    safe_name = get_safe_filename(subject_path)
    output_path = os.path.join(output_dir, f"{safe_name}.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path
    except OSError:
        pass
    fallback_name = f"processed_{int(time.time() * 1000)}.json"
    fallback_path = os.path.join(output_dir, fallback_name)
    try:
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return fallback_path
    except Exception as e:
        print(f"Failed all save methods: {e}")
        return None

async def progress_callback(message: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Progress: {message}", flush=True)

# --- Main ---
async def main(root_pdf, csv_path, output_dir, max_concurrent=10):
    df = pd.read_csv(csv_path)
    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(max_concurrent)
    already_processed = 0
    total_files = len(df)

    async def process_row(index, row):
        nonlocal already_processed
        async with sem:
            subject_path = os.path.join(root_pdf, "all_together", row["Sujet"])
            correction_path = os.path.join(root_pdf, "all_together", row["Correction"])

            if file_already_processed(subject_path, output_dir):
                already_processed += 1
                print(f"[SKIP] Already processed: {row['Sujet']}")
                return

            print(f"[PROCESSING] {row['Sujet']}")
            try:
                result = await full_pipeline_async(subject_path, correction_path, loop, progress_callback)
                saved_path = save_json(result, subject_path, output_dir)
                if saved_path:
                    print(f"[OK] Saved to {saved_path}")
                else:
                    print(f"[FAIL] Could not save {row['Sujet']}")
            except Exception as e:
                print(f"[ERROR] Failed on {row['Sujet']}: {e}")

    await asyncio.gather(*(process_row(i, row) for i, row in df.iterrows()))

    print(f"\n=== SUMMARY ===")
    print(f"Total files: {total_files}")
    print(f"Already processed: {already_processed}")
    print(f"Newly processed: {total_files - already_processed}")

if __name__ == "__main__":
    csv_input = "../datas/pdf_pairs.csv"
    output_folder = "../datas/results_json"
    root_pdf = r"..\pdf"
    os.makedirs(output_folder, exist_ok=True)
    asyncio.run(main(root_pdf, csv_input, output_folder, max_concurrent=10))

