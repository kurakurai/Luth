import pandas as pd 
import os 
import json 
import re
from collections import Counter
from ast import literal_eval

# Charger CSV source
df = pd.read_csv("/home/gad/kurakura/french-slm/src/data/bac-fr/datas/questions_combinees.csv")

cleaned_dir = "/home/gad/kurakura/french-slm/src/data/bac-fr/scripts/cleaned_json"
cleaned = os.listdir(cleaned_dir)
files_sorted = sorted(cleaned, key=lambda x: int(x.split('.')[0]))

# Fonction pour corriger les backslashes LaTeX
def fix_latex_backslashes(text):
    """
    Corrige les backslashes excessifs dans les expressions LaTeX
    """
    if not isinstance(text, str):
        return text
    
    # Remplacer les quadruples backslashes par des doubles
    text = re.sub(r'\\\\\\\\', r'\\\\', text)
    
    # Remplacer les triples backslashes par des simples (pour LaTeX)
    text = re.sub(r'\\\\\\', r'\\', text)
    
    # Patterns spécifiques pour les commandes LaTeX courantes
    latex_commands = [
        'text', 'mathbb', 'mathcal', 'mathrm', 'sin', 'cos', 'tan', 'log', 'ln',
        'frac', 'sqrt', 'sum', 'int', 'prod', 'lim', 'alpha', 'beta', 'gamma',
        'delta', 'epsilon', 'theta', 'lambda', 'mu', 'nu', 'pi', 'sigma', 'phi',
        'psi', 'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Pi', 'Sigma',
        'Phi', 'Psi', 'Omega', 'nearrow', 'searrow', 'leftarrow', 'rightarrow'
    ]
    
    for cmd in latex_commands:
        # Corriger \\\\command en \command
        text = re.sub(rf'\\\\\\\\{cmd}', rf'\\{cmd}', text)
        text = re.sub(rf'\\\\{cmd}', rf'\\{cmd}', text)
    
    return text

# Nettoyage des chaînes (version améliorée)
def fix_text(raw):
    # Supprimer les caractères de contrôle sauf \n et \t
    raw = ''.join(c for c in raw if ord(c) >= 32 or c in '\n\t')
    
    # Ne pas doubler les backslashes qui sont déjà corrects pour LaTeX
    # Seulement échapper les backslashes isolés qui ne sont pas des commandes LaTeX
    raw = re.sub(r'(?<!\\)\\(?![\\nt"bfr/a-zA-Z])', r'\\\\', raw)
    
    return raw

# Formatage lisible pour affichage brut (version améliorée)
def sanity(sample):
    if isinstance(sample, str):
        # Appliquer la correction des backslashes LaTeX
        return fix_latex_backslashes(sample)
    elif isinstance(sample, dict):
        # Appliquer récursivement aux valeurs du dictionnaire
        result = {}
        for k, v in sample.items():
            if isinstance(v, str):
                result[k] = fix_latex_backslashes(v)
            elif isinstance(v, (dict, list)):
                result[k] = sanity(v)
            else:
                result[k] = v
        return '\n'.join(f"{k} : {v}" for k, v in result.items())
    elif isinstance(sample, list):
        # Appliquer récursivement aux éléments de la liste
        fixed_items = []
        for item in sample:
            if isinstance(item, str):
                fixed_items.append(fix_latex_backslashes(item))
            elif isinstance(item, (dict, list)):
                fixed_items.append(sanity(item))
            else:
                fixed_items.append(str(item))
        return '\n'.join(fixed_items)
    return ''

# Fonction pour nettoyer récursivement un objet JSON
def clean_json_object(obj):
    """
    Nettoie récursivement tous les strings dans un objet JSON
    en corrigeant les backslashes LaTeX
    """
    if isinstance(obj, str):
        return fix_latex_backslashes(obj)
    elif isinstance(obj, dict):
        return {k: clean_json_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_object(item) for item in obj]
    else:
        return obj

# Statistiques d'échec
failure_stats = Counter()
ignored_files = []

# Parse robuste avec raisons d'échec (version améliorée)
def clean_json_file_with_reason(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read().strip()

    if not raw_text:
        failure_stats['vide'] += 1
        return {}, "vide"

    # Tentative brute
    try:
        obj = json.loads(raw_text)
        return clean_json_object(obj), None
    except:
        pass

    # Tentative avec correction d'antislashs
    try:
        obj = json.loads(fix_text(raw_text))
        return clean_json_object(obj), None
    except:
        pass

    # Cas de présence d'un champ "raw_output"
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict) and 'raw_output' in obj:
            nested_fixed = fix_text(obj['raw_output'])
            try:
                nested_obj = json.loads(nested_fixed)
                return clean_json_object(nested_obj), None
            except:
                failure_stats['raw_output_invalide'] += 1
                return {}, "raw_output_invalide"
    except:
        pass

    # Dernier recours : JSON mal écrit avec des guillemets simples
    try:
        obj = literal_eval(raw_text)
        if isinstance(obj, dict):
            return clean_json_object(obj), None
    except:
        failure_stats['literal_eval_fail'] += 1

    failure_stats['json_invalide_total'] += 1
    return {}, "json_invalide_total"

# Résultats
final_df = pd.DataFrame(columns=['conversations', 'subject', 'difficulty'])

template = lambda question, context, response: [
    {"role": "user", "content": context + "\n" + question + "\n"},
    {"role": "assistant", "content": response}
]

for i, row in df.iterrows():
    if i >= len(files_sorted):
        break

    file_path = os.path.join(cleaned_dir, files_sorted[i])
    print(f"⏳ Processing: {files_sorted[i]}")
    
    clean, reason = clean_json_file_with_reason(file_path)

    question = sanity(clean.get('question', ''))
    context = sanity(clean.get('context', ''))
    response = sanity(clean.get('reponse', ''))
    subject = sanity(clean.get('subject', ''))

    if not subject:
        failure_stats['subject_vide'] += 1
        ignored_files.append((files_sorted[i], 'subject_vide'))
        continue

    if not any([question.strip(), context.strip(), response.strip()]):
        failure_stats['champs_vides'] += 1
        ignored_files.append((files_sorted[i], 'champs_vides'))
        continue

    df.loc[i, 'questions'] = question
    df.loc[i, 'context'] = context
    df.loc[i, 'responses'] = response

    final_df.loc[i, 'conversations'] = template(question, context, response)
    final_df.loc[i, 'subject'] = subject
    final_df.loc[i, 'difficulty'] = 'medium'

# Export des résultats avec options pour préserver les backslashes
final_df.to_json("../datas/Scholar-fr-V0.json", orient="records", lines=True, force_ascii=False)

# Pour le CSV, utiliser des options qui préservent mieux les backslashes
final_df.to_csv("../datas/Scholar-fr-V0.csv", index=False, escapechar=None, quoting=1)

# Log des fichiers ignorés avec raison
with open("ignored_files.txt", "w") as f:
    for fname, reason in ignored_files:
        f.write(f"{fname}\t{reason}\n")

# Résumé
print(f"\n✅ Fichiers traités avec succès : {len(final_df)}")
print(f"❌ Total fichiers ignorés : {len(ignored_files)}")
print("\n📊 Statistiques des erreurs :")
for reason, count in failure_stats.items():
    print(f" - {reason} : {count}")

# Test de vérification
print("\n🔍 Test de vérification des backslashes LaTeX :")
test_strings = [
    "\\\\\\\\text{sn}(u, k)",
    "\\\\text{cn}(u, k)",
    "\\text{correct}",
    "\\\\\\nearrow"
]

for test in test_strings:
    fixed = fix_latex_backslashes(test)
    print(f"'{test}' → '{fixed}'")