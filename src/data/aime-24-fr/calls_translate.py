import os
import json
import asyncio
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from datasets import load_dataset,Dataset

# Initialise Vertex AI
PROJECT_ID = "elated-effect-466816-h9" #Put project ID
LOCATION = "us-central1" #Put zone
vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash")

def call_gemini(prompt):
    try:
        config = GenerationConfig(response_mime_type="application/json")
        response = model.generate_content(prompt)#generation_config=config
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini call failed: {e}")
        return None

def translate_aime(sample):
    prompt = f"""You are a professional technical translator. Translate the following text from english to french.
                Ensure absolute fidelity to:
                    Mathematical equations (keep all variables, LaTeX code, and formatting exactly as they are)\n
                    Technical terms and symbols \n
                    Logical structure and sentence integrity \n
                    Do not simplify, interpret, or omit any part of the content. Only translate natural language parts. \n
                    Format all equations, code, or inline symbols exactly as in the original. \n
                Here is the text to translate:
                    {sample}
                """
    return call_gemini(prompt)


def main():
    aime_fr = []
    dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]

    for i, sample in enumerate(dataset) : 
        print(f"processing sample: {i}")

        problem = translate_aime(sample['problem'])
        solution = translate_aime(sample['solution'])
        
        aime_fr.append({'id':sample['id'],
                        'problem':problem,
                        'solution':solution,
                        'answer':sample['answer'],
                        'url':sample['url'],
                        'year':sample['year']})

    final_dataset = Dataset.from_list(aime_fr)
    final_dataset.save_to_disk("aime_2024_fr")
    final_dataset.push_to_hub("kurakurai/aime_2024_fr")

if __name__=="__main__":
    main()


