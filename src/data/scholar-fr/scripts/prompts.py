from enum import Enum
import json

class Prompts(Enum):

    extract_questions = lambda subject_text: f"""
        You are given the full text of a French high school exam.

        Task:
        - Extract all actual **student questions** that require an answer.
        - Do not include titles, sections, instructions, or metadata.
        - Questions can be numbered in many ways (1., 1), 1a, Q1, etc.) — detect all valid forms.
        - All equations should be reformulated in LaTeX format with $ prefix and suffix.
        - Return a **JSON list of strings**: one entry per question.
        - Do not rephrase. Copy questions exactly as they appear.

        Exam text:
        {subject_text}
    """

    filter_questions = lambda joined_questions: f"""
        You are given a list of French high school exam questions. 
        Return **only** the ones that can be answered **without any diagram, image, drawing, or graph**. 
        List them exactly as they appear. Do not explain or rephrase.

        List:
        {joined_questions}
    """

    add_context_to_questions = lambda subject_text, questions: f"""
        The following is the full text of a French high school exam:
        ----
        {subject_text}
        ----

        For each question below, extract its introductory description from the subject (such as problem description or setup).
        Do not summarize or rewrite. 
        Return a JSON list of dictionaries with keys: "question" and "context".

        Questions:
        {json.dumps(questions, ensure_ascii=False) if isinstance(questions, list) else questions}
    """

    extact_answers = lambda contexted_questions, correction_text: f"""
        You are extracting exact answers (in French) from the correction of a French high school exam.

        Rules:
        - Use only what is explicitly written in the correction text below.
        - Return the answer **exactly** as it appears, in French.
        - All equations should be reformulated in LaTeX format with $ prefix and suffix.

        Correction:
        {correction_text}

        Questions with context:
        {json.dumps(contexted_questions, ensure_ascii=False) if isinstance(contexted_questions, list) else contexted_questions}
    """

    enrich_context = lambda answers_with_context: f"""
        Some questions below may require previous answers to make sense (multi-step problems).

        For each question, if a previous answer is needed to understand or solve it, add it to the context.

        Return a JSON list of updated items with keys: "question", "context", "reponse"

        Entries:
        {json.dumps(answers_with_context, ensure_ascii=False) if isinstance(answers_with_context, list) else answers_with_context}
    """

    correct_json_list = lambda json_malformed: f"""
        You are a JSON formatting assistant. I will provide you with a malformed JSON list that contains several entries, each representing a question-answer pair related to mathematics. Each entry should have the following structure:

        {{
        "question": "...",
        "context": "...",  // this can be null or empty if there's no context
        "reponse": "..."
        }}

        Please:
        - Parse the malformed JSON correctly.
        - Fix any incorrect formatting (e.g. escape characters, newline issues, incorrect indentation).
        - Ensure that each item is a valid JSON object with the three keys: "question", "context", "reponse".
        - Make sure the final output is a valid, well-formatted JSON array that can be parsed by standard JSON parsers.
        - The entire output must be in French.
        Do not change the content of the "question", "context", or "reponse" fields — only clean formatting and structure.

        Here is the malformed JSON input:
        {json_malformed}
    """

    correct_json_dict = lambda json_malformed: f"""
        You are a JSON formatting assistant. I will provide you with a malformed JSON dict that contains three pairs of keys and values, representing a question-answer pair related to mathematics. It should have the following structure:

        {{
        "question": "...",
        "context": "...",  // this can be null or empty if there's no context
        "reponse": "..."
        }}

        Please:
        - Parse the malformed JSON correctly.
        - Fix any incorrect formatting (e.g. escape characters, newline issues, incorrect indentation).
        - Ensure the item is a valid JSON object with the three keys: "question", "context", "reponse".
        - Make sure the final output is a valid, well-formatted JSON dict that can be parsed by standard JSON parsers.
        - The entire output must be in French.
        Do not change the content of the "question", "context", or "reponse" fields — only clean formatting and structure.

        Here is the malformed JSON input:
        {json_malformed}
    """

    clean_sample = lambda question, context, response: f"""
        You will receive three inputs: a question, a context, and an answer.

        Your tasks are:
        - Correct any errors in spelling, grammar, LaTeX, and formatting in all three inputs.
        - Carefully review the context and correct it if there are any issues. If the context is missing or empty but should be present based on the question and answer, generate a relevant and useful context.
        - Translate all English text fully into French.
        - Rephrase the answer to add clarity by:
            - Expanding on the reasoning,
            - Breaking the answer down into logical steps or explanations,
            - Justifying the conclusion.
        The final answer must remain logically and factually equivalent to the original.
        - Do not change the overall intent of the question, context, or response unless necessary for correction.
        - Provide your output strictly in JSON format with the following keys: "question", "context", and "reponse".
        - The entire output must be in French.
        - Do not add any hints in the context related to the question.
        - If the question content is missing (i.e., only the question number is present), generate a relevant and useful question according to the context and question.
        Entries:
        Question: {question}
        Context: {context}
        Response: {response}
        """
