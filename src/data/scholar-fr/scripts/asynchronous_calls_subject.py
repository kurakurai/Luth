import os
import json
import asyncio
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Initialise Vertex AI
PROJECT_ID = "" #Put project ID
LOCATION = "" #Put zone
vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash-lite")

async def call_gemini_subject(prompt, loop):
    def call_sync():
        try:
            config = GenerationConfig(response_mime_type="application/json")
            response = model.generate_content(prompt, generation_config=config)
            return response.text.strip()
        except Exception as e:
            print(f"[ERROR] Gemini call failed: {e}")
            return None
    return await loop.run_in_executor(None, call_sync)

def get_safe_subject_json(raw_response):
    try:
        data = json.loads(raw_response)
        if 'subject' in data and data['subject'] in ['maths', 'physics-chemistry']:
            return data['subject']
        else:
            print(f"[WARN] Subject key missing or unexpected value in response: {raw_response}")
            return None
    except Exception as e:
        print(f"[WARN] Failed to parse JSON subject: {e}")
        return None

async def update_subject_for_file(file_path, loop):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        return

    question_text = data.get("question", "") or data.get("questions", "")
    context_text = data.get("context", "")

    prompt = (
        "Détermine la matière principale de la question ci-dessous. "
        "Répond uniquement en JSON avec la clé 'subject' dont la valeur est soit 'maths', soit 'physics-chemistry'.\n\n"
        f"Contexte:\n{context_text}\n\n"
        f"Question:\n{question_text}\n\n"
        "Réponse :"
    )

    raw_response = await call_gemini_subject(prompt, loop)

    if not raw_response:
        print(f"[ERROR] No response for {file_path}")
        return

    subject = get_safe_subject_json(raw_response)
    if not subject:
        print(f"[ERROR] Unrecognized subject for {file_path}, raw response: {raw_response}")
        return

    data["subject"] = subject

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[UPDATED] Subject added in {os.path.basename(file_path)}: {subject}")
    except Exception as e:
        print(f"[ERROR] Failed to save {file_path}: {e}")

async def main_subject_classification(cleaned_dir, max_concurrent=5):
    loop = asyncio.get_event_loop()

    files = [f for f in os.listdir(cleaned_dir) if f.endswith(".json")]
    total = len(files)
    print(f"Number of files to process: {total}")

    sem = asyncio.Semaphore(max_concurrent)  # Limite la concurrence

    async def sem_task(file_name):
        async with sem:
            await update_subject_for_file(os.path.join(cleaned_dir, file_name), loop)

    tasks = [sem_task(f) for f in files]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    cleaned_json_folder = "cleaned_json"
    max_concurrent_requests = 1 
    asyncio.run(main_subject_classification(cleaned_json_folder, max_concurrent=max_concurrent_requests))
