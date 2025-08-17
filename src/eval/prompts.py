import random
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.requests import Doc


# Ifeval-fr prompt function
def prompt_ifeval_fr(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        specific={
            "instructions_id_list": line["instruction_id_list"],
            "kwargs": line["kwargs"],
        },
    )


# gpqa-diamond-fr prompt function
def gpqa_fr_instruct(line, task_name: str = None):
    """Prompt template adapted from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14"""
    gold_index = random.randint(0, 3)
    choices = [
        line["Incorrect Answer 1"],
        line["Incorrect Answer 2"],
        line["Incorrect Answer 3"],
    ]
    choices.insert(gold_index, line["Correct Answer"])
    query_template = (
        "Répondez à la question à choix multiple suivante. "
        "La dernière ligne de votre réponse doit avoir le format suivant : 'Réponse : $LETTRE' "
        "(sans les guillemets) où LETTRE est l'une de A, B, C ou D. "
        "Réfléchissez étape par étape avant de répondre.\n\n"
        "{Question}\n\n"
        "A) {A}\n"
        "B) {B}\n"
        "C) {C}\n"
        "D) {D}"
    )
    query = query_template.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["problem"]
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
    )


# hellaswag-fr prompt function
def prompt_hellaswag_fr(line, task_name: str = None):
    query = "Voici une série de questions à choix multiples (avec réponses) pour évaluer le bon sens.\n\n"
    query += f"Question: {line['ctx_a']} {line['ctx_b'].capitalize()}\n"
    query += "".join(
        [f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["endings"])]
    )
    query += "Réponse:"

    gold_ix = int(line["label"]) if line["label"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + i for i in LETTER_INDICES[: len(line["endings"])]],
        gold_index=gold_ix,  # -1 pour le test
    )


# boolq-fr prompt function
def prompt_boolq_fr(line, task_name: str = None):
    question = (
        line["question"][:-1] if line["question"][-2:] == "??" else line["question"]
    )
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {question}\nRéponse: ",
        choices=["Non", "Oui"],
        gold_index=int(line["label"]),
    )


# bbh-fr prompt functions
def bbh(line, instruction, choices, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{instruction}Q: {line['input']}\nA:",
        choices=choices,
        gold_index=choices.index(line["target"]),
    )


def bbh_boolean_expressions(line, task_name: str = None):
    instruction = "Évalue le résultat d'une expression booléenne aléatoire.\n\n"
    choices = ["Incorrect", "Vrai"]
    return bbh(line, instruction, choices, task_name)


def bbh_causal_judgment(line, task_name: str = None):
    instruction = "Réponds à des questions sur l’attribution causale.\n\n"
    choices = ["Oui", "Non"]
    return bbh(line, instruction, choices, task_name)


def bbh_date_understanding(line, task_name: str = None):
    instruction = "Infère la date à partir du contexte.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:6]]
    return bbh(line, instruction, choices, task_name)


def bbh_disambiguation_qa(line, task_name: str = None):
    instruction = "Clarifie le sens des phrases avec des pronoms ambigus.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:3]]
    return bbh(line, instruction, choices, task_name)


def bbh_dyck_languages(line, task_name: str = None):  # Can only be done in generative
    instruction = "Ferme correctement un mot Dyck-n.\n\n"
    choices = [line["target"]]
    return bbh(line, instruction, choices, task_name)


def bbh_formal_fallacies(line, task_name: str = None):
    instruction = (
        "Distingue les arguments déductivement valides des sophismes formels.\n\n"
    )
    choices = ["valide", "invalidee"]
    return bbh(line, instruction, choices, task_name)


def bbh_geometric_shapes(line, task_name: str = None):
    instruction = "Nomme les formes géométriques à partir de leurs chemins SVG.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:11]]
    return bbh(line, instruction, choices, task_name)


def bbh_hyperbaton(line, task_name: str = None):
    instruction = "Ordonne correctement les adjectifs dans des phrases anglaises.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:2]]
    return bbh(line, instruction, choices, task_name)


def bbh_logical_deduction_five_objects(line, task_name: str = None):
    instruction = "Une tâche de déduction logique qui nécessite de déduire l’ordre d’une séquence d’objets.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:5]]
    return bbh(line, instruction, choices, task_name)


def bbh_logical_deduction_seven_objects(line, task_name: str = None):
    instruction = "Une tâche de déduction logique qui nécessite de déduire l’ordre d’une séquence d’objets.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:7]]
    return bbh(line, instruction, choices, task_name)


def bbh_logical_deduction_three_objects(line, task_name: str = None):
    instruction = "Une tâche de déduction logique qui nécessite de déduire l’ordre d’une séquence d’objets.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:3]]
    return bbh(line, instruction, choices, task_name)


def bbh_movie_recommendation(line, task_name: str = None):
    if line["target"] == "Monsters, Inc":  # this line is not correctly formatted
        print(
            "One sample removed from task bbh:movie_recommendation because its line is incorrectly formatted."
        )
        return []
    instruction = "Recommande des films similaires à la liste de films donnée.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:6]]
    return bbh(line, instruction, choices, task_name)


def bbh_multistep_arithmetic_two(line, task_name: str = None):
    instruction = "Résous des problèmes arithmétiques à étapes multiples.\n\n"  # Can only be done in generative
    choices = [line["target"]]
    return bbh(line, instruction, choices, task_name)


def bbh_navigate(line, task_name: str = None):
    instruction = "Étant donné une série d’instructions de navigation, détermine si l’on revient au point de départ.\n\n"
    choices = ["Oui", "Non"]
    return bbh(line, instruction, choices, task_name)


def bbh_object_counting(line, task_name: str = None):
    instruction = "Questions impliquant de dénombrer des objets et demander au modèle de les compter.\n\n"
    choices = [str(i) for i in range(1, 19)]
    return bbh(line, instruction, choices, task_name)


def bbh_penguins_in_a_table(line, task_name: str = None):
    instruction = (
        "Réponds à des questions sur un tableau de pingouins et leurs attributs.\n\n"
    )
    choices = [f"({c})" for c in LETTER_INDICES[:5]]
    return bbh(line, instruction, choices, task_name)


def bbh_reasoning_about_colored_objects(line, task_name: str = None):
    instruction = "Réponds à des questions très simples sur les couleurs des objets sur une surface.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:18]]
    return bbh(line, instruction, choices, task_name)


def bbh_ruin_names(line, task_name: str = None):
    if line["target"] in [
        "dearth, wind, & fire",
        "rita, sue and bob poo",
    ]:  # line not correctly formatted
        print(
            "One sample removed from task bbh:ruin_names because its line is incorrectly formatted."
        )
        return []
    instruction = "Choisis la modification humoristique qui « ruine » le nom du film ou de l’artiste musical donné.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:6]]
    return bbh(line, instruction, choices, task_name)


def bbh_salient_translation_error_detection(line, task_name: str = None):
    instruction = "Détecte le type d’erreur dans une traduction anglaise d’une phrase source allemande.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:6]]
    return bbh(line, instruction, choices, task_name)


def bbh_snarks(line, task_name: str = None):
    instruction = "Détermine laquelle des deux phrases est sarcastique. Selon le dictionnaire de l’université de Cambridge, le sarcasme est « l’utilisation de remarques qui signifient clairement le contraire de ce qu’elles disent, faites pour blesser quelqu’un ou critiquer quelque chose de manière humoristique ». Les phrases sarcastiques contiennent souvent des propos satiriques ou ironiques, des hyperboles, des remarques ambivalentes ou spirituelles.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:2]]
    return bbh(line, instruction, choices, task_name)


def bbh_sports_understanding(line, task_name: str = None):
    instruction = "Détermine si une phrase artificiellement construite liée au sport est plausible ou non.\n\n"
    choices = ["Oui", "Non"]
    return bbh(line, instruction, choices, task_name)


def bbh_temporal_sequences(line, task_name: str = None):
    instruction = "Description de tâche: Réponds à des questions sur les moments auxquels certains événements ont pu se produire.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:4]]
    return bbh(line, instruction, choices, task_name)


def bbh_tracking_shuffled_objects_five_objects(line, task_name: str = None):
    instruction = "Détermine les positions finales d’un ensemble d’objets données leurs positions initiales et la description d’une séquence d’échanges.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:5]]
    return bbh(line, instruction, choices, task_name)


def bbh_tracking_shuffled_objects_seven_objects(line, task_name: str = None):
    instruction = "Détermine les positions finales d’un ensemble d’objets données leurs positions initiales et la description d’une séquence d’échanges.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:7]]
    return bbh(line, instruction, choices, task_name)


def bbh_tracking_shuffled_objects_three_objects(line, task_name: str = None):
    instruction = "Détermine les positions finales d’un ensemble d’objets données leurs positions initiales et la description d’une séquence d’échanges.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:3]]
    return bbh(line, instruction, choices, task_name)


def bbh_web_of_lies(line, task_name: str = None):
    instruction = "Évalue une fonction booléenne aléatoire exprimée sous forme de problème en mots.\n\n"
    choices = ["Oui", "Non"]
    return bbh(line, instruction, choices, task_name)


def bbh_word_sorting(line, task_name: str = None):
    instruction = "Trie une liste de mots.\n\n"  # Can only be done in generative
    choices = [line["target"]]
    return bbh(line, instruction, choices, task_name)


# musr-fr prompt function
def musr_fr(line, task_name: str = None):
    choices = line["choices"]
    query = line["narrative"] + "\n\n"
    query += line["question"] + "\n\n"
    for i, choice in enumerate(choices):
        query += f"{i + 1} - {choice}\n"
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=line["answer_index"],
    )


def prompt_math_500_fr(line, task_name: str = None):
    MATH_QUERY_TEMPLATE = """
Résous le problème de mathématiques suivant de manière efficace et claire. La dernière ligne de ta réponse doit être au format suivant : 'Donc, la réponse finale est : $\\boxed{{RÉPONSE}}$. J’espère que c’est correct' (sans les guillemets), où RÉPONSE est simplement le nombre ou l’expression qui résout le problème. Raisonne étape par étape avant de répondre.

{Question}
""".strip()
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        gold_index=0,
        choices=[line["solution"]],
    )


def prompt_aime_fr(line, task_name: str = None):
    MATH_QUERY_TEMPLATE = """
Résous clairement et efficacement le problème de mathématiques suivant. La dernière ligne de ta réponse doit être au format suivant : 'Donc, la réponse finale est : $\\boxed{{RÉPONSE}}$. J’espère que c’est correct' (sans les guillemets), où RÉPONSE est simplement le nombre ou l’expression finale qui résout le problème. Réfléchis étape par étape avant de répondre.

{Question}
""".strip()
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def prompt_arc_fr(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]],
        gold_index=["A", "B", "C", "D"].index(line["answerKey"]),
    )


# math-fr prompt function
NL_PROMPT = """Problème:
Déterminer le domaine de l'expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.

Solution:
Les expressions sous chaque racine carrée doivent être positives ou nulles. Ainsi, $x - 2 \\geq 0$, donc $x \\geq 2$, et $5 - x \\geq 0$, donc $x \\leq 5$. De plus, le dénominateur ne peut pas être nul, donc $5 - x > 0$, ce qui donne $x < 5$. Par conséquent, le domaine de l'expression est $\\boxed{[2,5)}$.
Réponse finale : La réponse finale est $[2,5)$.

Problème:
Si $\\det \\mathbf{A} = 2$ et $\\det \\mathbf{B} = 12$, déterminer $\\det (\\mathbf{A} \\mathbf{B})$.

Solution:
On a $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}$.
Réponse finale : La réponse finale est $24$.

Problème:
Terrell soulève habituellement deux haltères de 20 livres 12 fois. S'il utilise plutôt deux haltères de 15 livres, combien de fois doit-il les soulever pour soulever le même poids total ?

Solution:
Si Terrell soulève deux haltères de 20 livres 12 fois, il soulève un total de $2 \\cdot 12 \\cdot 20 = 480$ livres. S'il soulève plutôt deux haltères de 15 livres $n$ fois, il soulèvera un total de $2 \\cdot 15 \\cdot n = 30n$ livres. En égalant cela à 480 livres, on peut résoudre pour $n$ :
\\begin{align*}
30n &= 480\\\\
\\Rightarrow\\qquad n &= 480/30 = \\boxed{16}
\\end{align*}
Réponse finale : La réponse finale est $16$.

Problème:
Si le système d'équations suivant

\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b
\\end{align*}

admet une solution $(x, y)$ avec $x$ et $y$ tous deux non nuls, déterminer $\\frac{a}{b}$, en supposant que $b$ est non nul.

Solution:
Si on multiplie la première équation par $-\\frac{3}{2}$, on obtient

$$6y - 9x = -\\frac{3}{2}a.$$

Or on sait aussi que $6y - 9x = b$, donc

$$-\\frac{3}{2}a = b \\Rightarrow \\frac{a}{b} = \\boxed{-\\frac{2}{3}}.$$

Réponse finale : La réponse finale est $-\\frac{2}{3}$.
"""


def prompt_math_fr(line, task_name: str = None):
    """
    Prompt function for the Math-Fr task. With few-shot examples.
    """
    return Doc(
        task_name=task_name,
        query=f"{NL_PROMPT}\n\nProblem:\n{line['problem']}\n\nSolution:",
        choices=[line["solution"]],
        gold_index=0,
    )
