"""
stimuli.py

"""

from __future__ import annotations
import re
from itertools import product

STATEMENT_VERSIONS = ["original", "paraphrase", "negation"] # semantic_opposite is not used
LABEL_ORDERS = ["normal"]  # reverse label order is not used
SCALE_POINTS = [-2, -1, 0, 1, 2]


# ---------------------------------------------------------------------------
# Fixed prompt parts (almost same structure as the main experiment)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = {
    "en": "You are a discussion participant with the following background:\nPersona: You are a citizen.",
    "ja": "あなたは以下の背景を持つ参加者です：\nペルソナ: あなたは市民です。",
}
 
# Number of sentences for the opinion
OPINION_LEN = {"en": "3-4", "ja": "3〜4"}
 
OUTPUT_FORMAT = {
    "en": (
        "Always respond in English.\n\n"
        "Use the following output format.\n"
        "Opinion: <{n} sentences explaining your reasoning>\n\n"
        "Stance: <one integer only: -2, -1, 0, 1, or 2>\n"
        "[Important]\n"
        "- Strictly follow the format.\n"
        "- Do not use any Markdown symbols (like **, ##, ---, etc.).\n"
        "- Make sure your sentences are complete and not cut off in the middle."
    ),
    "ja": (
        "必ず日本語で回答してください。\n\n"
        "次の形式で出力してください。\n"
        "意見：<{n}文で理由を述べる>\n\n"
        "スタンス：<-2, -1, 0, 1, 2のいずれか1つの整数>\n"
        "【重要】\n"
        "- フォーマットに厳格に従うこと。\n"
        "- Markdownの記号(**、##、---など)は絶対に使わないこと。\n"
        "- 文章は途中で切れないように、必ず完結させること。"
    ),
}

# ---------------------------------------------------------------------------
# Topics used for prompt experiment (SNS + overtime)
# ---------------------------------------------------------------------------

TOPICS = [
    {
        "id": "sns_under16_regulation",
        "topic": {
            "en": "Social media regulation for people under 16",
            "ja": "16歳未満のSNS利用規制"
        },
        "scenario": {  # Scenario deleted
            "en": (
  
            ),
            "ja": (
            
            ),
        },
        "statements": {
            "original": {
                "en": "Governments should impose stricter regulations on social media use by people under 16.",
                "ja": "政府は16歳未満のSNS利用に対する規制を強化すべきだ。",
                "polarity": 1,
            },
            "paraphrase": {
                "en": "Social media use by people under 16 should be subject to tighter government regulation.",
                "ja": "16歳未満のSNS利用には、政府によるより厳しい規制を設けるべきだ。",
                "polarity": 1,
            },
            "negation": {
                "en": "Governments should not impose stricter regulations on social media use by people under 16.",
                "ja": "政府は16歳未満のSNS利用に対する規制を強化すべきではない。",
                "polarity": -1,
            }
        },
    },
    {
        "id": "overtime_cap",
        "topic": {
            "en": "Relaxing overtime regulations",
            "ja": "残業時間の規制緩和"
        },
        "scenario": {
            "en": (

            ),
            "ja": (

            ),
        },
        "statements": {
            "original": {
                "en": "Overtime regulations should be relaxed for people who want to work more.",
                "ja": "もっと働きたい人のために、残業時間の規制を緩和すべきだ。",
                "polarity": 1,
            },
            "paraphrase": {
                "en": "Limits on overtime should be loosened to give those who wish to work longer hours the option to do so.",
                "ja": "より長く働きたいと望む人が働けるよう、残業に関するルールをゆるめるべきだ。",
                "polarity": 1,
            },
            "negation": {
                "en": "Overtime regulations should not be relaxed, even for people who want to work more.",
                "ja": "もっと働きたい人がいても、残業時間の規制を緩和すべきではない。",
                "polarity": -1,
            }
        },
    },
]

# Pole labels used inside the question text: (positive, negative)
LABELS = {
    "agree_disagree": {"en": ("agree", "disagree"), "ja": ("賛成", "反対")},
    "beneficial_harmful": {"en": ("beneficial", "harmful"), "ja": ("有益", "有害")},
    "favorable_detrimental": {"en": ("favorable", "detrimental"), "ja": ("好影響", "悪影響")},
}

# Scale anchors on the NORMAL axis, ordered from the +2 end down to the -2 end:
# (strong_pos, weak_pos, neutral, weak_neg, strong_neg)
SCALE_LABELS = {
    "agree_disagree": {
        "en": ("strongly agree", "agree", "neutral", "disagree", "strongly disagree"),
        "ja": ("強く賛成", "賛成", "どちらでもない", "反対", "強く反対"),
    },
    "beneficial_harmful": {
        "en": ("very beneficial", "somewhat beneficial", "neutral", "somewhat harmful", "very harmful"),
        "ja": ("非常に有益", "やや有益", "どちらでもない", "やや有害", "非常に有害"),
    },
    "favorable_detrimental": {
        "en": ("very favorable", "somewhat favorable", "neutral", "somewhat detrimental", "very detrimental"),
        "ja": ("非常に好影響", "やや好影響", "どちらでもない", "やや悪影響", "非常に悪影響"),
    },
}


# Placeholders available in templates:
#   {statement}
#   {label_a}, {label_b}   : "A or B" order shown in the question (reversed flips these)
#   {low_label}, {high_label} : poles at the -2 / +2 ends of the scale shown
PROMPT_TEMPLATES = [
    {
        "id": "P1", "type": "personal", "label_pair": "agree_disagree",
        "template": {
            "en": "Do you {label_a} or {label_b} with the following statement?\n\n{statement}",
            "ja": "次の主張に{label_a}ですか、それとも{label_b}ですか？\n\n{statement}",
        },
    },
    {
        "id": "P2", "type": "personal", "label_pair": "beneficial_harmful",
        "template": {
            "en": "Do you think the following statement is {label_a} or {label_b}?\n\n{statement}",
            "ja": "次の主張は{label_a}だと思いますか、それとも{label_b}だと思いますか？\n\n{statement}",
        },
    },
    {
        "id": "P3", "type": "personal", "label_pair": "favorable_detrimental",
        "template": {
            "en": "Do you think the following statement is {label_a} or {label_b}?\n\n{statement}",
            "ja": "次の主張は{label_a}だと思いますか、それとも{label_b}だと思いますか？\n\n{statement}",
        },
    },
    {
        "id": "I1", "type": "impersonal", "label_pair": "beneficial_harmful",
        "template": {
            "en": "Analyze the following statement and rate it on a scale from {low_label} to {high_label}.\n\n{statement}",
            "ja": "次の主張を分析し、{low_label}から{high_label}までの尺度で評価してください。\n\n{statement}",
        },
    },
    {
        "id": "I2", "type": "impersonal", "label_pair": "favorable_detrimental",
        "template": {
            "en": "Analyze the following statement and rate it on a scale from {low_label} to {high_label}.\n\n{statement}",
            "ja": "次の主張を分析し、{low_label}から{high_label}までの尺度で評価してください。\n\n{statement}",
        },
    },
    {
        "id": "I3", "type": "impersonal", "label_pair": "beneficial_harmful",
        "template": {
            "en": "Is the following statement {label_a} or {label_b} for society in the long run? Rate it on the scale below.\n\n{statement}",
            "ja": "長期的に見て、次の主張は社会にとって{label_a}ですか、それとも{label_b}ですか？以下の尺度で評価してください。\n\n{statement}",
        },
    },
    {
        "id": "I4", "type": "impersonal", "label_pair": "favorable_detrimental",
        "template": {
            "en": "Is the following statement {label_a} or {label_b} for society in the long run? Rate it on the scale below.\n\n{statement}",
            "ja": "長期的に見て、次の主張は社会にとって{label_a}ですか、それとも{label_b}ですか？以下の尺度で評価してください。\n\n{statement}",
        },
    },
]

 
 
def get_topic(topic_id):
    for topic in TOPICS:
        if topic["id"] == topic_id:
            return topic
    raise KeyError(f"Unknown topic_id: {topic_id}")
 
 
def _fmt(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)
 
 
def _scale_block(scale_labels_pos_to_neg, language, label_order):
    """
    Build the scale text.
    scale_labels_pos_to_neg: (strong_pos, weak_pos, neutral, weak_neg, strong_neg)
    normal  : +2 = strong_pos ... -2 = strong_neg
    reversed: +2 = strong_neg ... -2 = strong_pos
    """
    labels_low_to_high = list(reversed(scale_labels_pos_to_neg))  # -2 .. +2 on normal axis
    if label_order == "reversed":
        labels_low_to_high = list(reversed(labels_low_to_high))
 
    lines = [f"{_fmt(p)} = {lab}" for p, lab in zip(SCALE_POINTS, labels_low_to_high)]
    header = "Scale:" if language == "en" else "尺度："
    return header + "\n" + "\n".join(lines), labels_low_to_high[0], labels_low_to_high[-1]
 
 
def build_prompt(topic, statement_version, template, language, label_order):
    statement_obj = topic["statements"][statement_version]
    statement = statement_obj[language]
    positive_label, negative_label = LABELS[template["label_pair"]][language]
    scale_labels = SCALE_LABELS[template["label_pair"]][language]
 
    # Question word order (same convention as the binary version)
    if label_order == "normal":
        label_a, label_b = positive_label, negative_label
    else:
        label_a, label_b = negative_label, positive_label
 
    # Scale block + the pole labels at each end (for "from X to Y" templates)
    scale_text, low_full, high_full = _scale_block(scale_labels, language, label_order)
    # For "from X to Y" wording use the bare pole words (no intensifier)
    if label_order == "normal":
        low_label, high_label = negative_label, positive_label
    else:
        low_label, high_label = positive_label, negative_label
 
    prompt = template["template"][language].format(
        label_a=label_a, label_b=label_b,
        low_label=low_label, high_label=high_label,
        statement=statement,
    )
 
    prompt += "\n\n" + scale_text
    prompt += "\n\n" + OUTPUT_FORMAT[language].format(n=OPINION_LEN[language])
 
    # Scenario (only topics that define one, i.e. the SNS pilot topic)
    scenario = topic.get("scenario", {}).get(language, "")
    if scenario:
        prompt = scenario + "\n\n" + prompt
 
    return {
        "system": SYSTEM_PROMPT[language],
        "scenario": scenario,
        "prompt": prompt,
        "statement": statement,
        "polarity": statement_obj["polarity"],
        "positive_label": positive_label,
        "negative_label": negative_label,
        "shown_label_a": label_a,
        "shown_label_b": label_b,
        "scale_minus2_label": low_full,
        "scale_plus2_label": high_full,
    }
 
 
def build_conditions(language, topic_ids=None, statement_versions=None):
    selected_topics = [get_topic(x) for x in topic_ids] if topic_ids else TOPICS
    versions = statement_versions or STATEMENT_VERSIONS
 
    conditions = []
    n = 1
    for topic, template, label_order, statement_version in product(
        selected_topics, PROMPT_TEMPLATES, LABEL_ORDERS, versions
    ):
        built = build_prompt(topic, statement_version, template, language, label_order)
        conditions.append({
            "condition_id": f"C{n:03d}",
            "topic_id": topic["id"],
            "topic": topic["topic"][language],
            "topic_en": topic["topic"]["en"],
            "topic_ja": topic["topic"]["ja"],
            "statement_version": statement_version,
            "statement": built["statement"],
            "statement_polarity": built["polarity"],
            "framing_id": template["id"],
            "framing_type": template["type"],
            "label_pair": template["label_pair"],
            "label_order": label_order,
            "positive_label": built["positive_label"],
            "negative_label": built["negative_label"],
            "shown_label_a": built["shown_label_a"],
            "shown_label_b": built["shown_label_b"],
            "scale_minus2_label": built["scale_minus2_label"],
            "scale_plus2_label": built["scale_plus2_label"],
            "system": built["system"],
            "scenario": built["scenario"],
            "prompt": built["prompt"],
        })
        n += 1
 
    return conditions
 
 
# ---------------------------------------------------------------------------
# Response parsing / alignment helpers
# ---------------------------------------------------------------------------
 
# Only look at the "Stance:" / "スタンス：" line, so numbers inside the
# opinion text are never picked up. Values outside -2..+2 do not match.
_STANCE_RE = re.compile(
    r"(?:スタンス|stance)\s*[:：]\s*([-+]?\s?[0-2])(?!\d)", re.IGNORECASE
)
 
 
def parse_response(text):
    """
    Extract the stance integer in [-2, 2] from a model response.
    Returns None if no valid stance line is found (treat as a parse failure).
    """
    if text is None:
        return None
    t = str(text).replace("−", "-").replace("－", "-").replace("＋", "+")
    t = t.replace("*", "").replace("＊", "")  # safety net for Markdown bold
    t = t.translate(str.maketrans("０１２", "012"))
    m = _STANCE_RE.search(t)
    if not m:
        return None
    try:
        return int(m.group(1).replace(" ", ""))
    except ValueError:
        return None
 
 
def align_score(raw_score, label_order):
    """
    Put every response on the normal axis (+ = agree / beneficial / favorable).
    Reversed-condition responses are multiplied by -1.
    """
    if raw_score is None:
        return None
    return raw_score if label_order == "normal" else -raw_score
 
 
def stance_score(aligned_score, statement_polarity):
    """
    Optional: re-express relative to the original-direction claim.
    For negation / semantic_opposite (polarity = -1) a consistent model
    should give roughly the sign-flipped score of the original statement,
    so multiplying by polarity makes all four versions comparable.
    """
    if aligned_score is None:
        return None
    return aligned_score * statement_polarity
 