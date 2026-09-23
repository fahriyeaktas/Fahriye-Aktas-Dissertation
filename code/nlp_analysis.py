"""
nlp_analysis.py
================================================================
Constructs all text-based communication measures for Prepared Remarks,
Q&A, and the whole conference:

  1. CentralBankRoBERTa sentiment (primary)
  2. Loughran-McDonald uncertainty (primary)
  3. FinBERT-FOMC sentiment (robustness)
  4. Hawkish/dovish stance (robustness, a distinct construct from sentiment)

The three sentiment measures (CentralBankRoBERTa, FinBERT-FOMC, and
the hawkish/dovish stance's underlying dictionary count) each use their
own aggregation procedure appropriate to their method: the two
transformer-based sentiment models use sentence-level, token-weighted
aggregation (see build_document_scorer); the hawkish/dovish stance uses
sentence-part phrase matching (see Section 4 below). Loughran-McDonald
uncertainty is a simple word-count ratio (1000 x uncertainty words /
total words) and is NOT token-weighted or sentence-aggregated in the
same sense - it is computed once over the full section text.
Whole-conference ("Overall") scores are
computed by concatenating the retained Prepared Remarks and Q&A text
and rescoring the combined text as ONE document — never by averaging
the two section-level scores.

Generic FinBERT (ProsusAI/finbert) was tested as a candidate primary
model on an eight-sentence macro-language sanity check and rejected in
favour of CentralBankRoBERTa (1/8 vs 8/8 correct). It is not part of
the final pipeline and is not reproduced here.
================================================================
"""
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
import pandas as pd

from data_cleaning import DATA_DIR, OUTPUT_DIR, load_transcript_files

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

LM_DICTIONARY_PATH = DATA_DIR / "Loughran-McDonald_MasterDictionary.csv"
TEXT_MEASURES_FILE = OUTPUT_DIR / "text_measures.xlsx"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ================================================================
# Shared sentence-level, token-weighted document scorer
# ================================================================

def build_document_scorer(tokenizer, model, id2label):
    """Returns a scorer function: Sentiment = P(positive) - P(negative),
    aggregated across sentences and 480-token chunks, weighted by each
    chunk's own token count."""

    def score_piece(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        return {id2label[i]: probs[i].item() for i in range(len(id2label))}

    def score_document(text):
        sentences = sent_tokenize(text)
        w_pos = w_neg = 0.0
        total_tok = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            tokens = tokenizer.tokenize(s)
            if not tokens:
                continue
            chunks = [tokens[i:i + 480] for i in range(0, len(tokens), 480)]
            for ch in chunks:
                ch_text = tokenizer.convert_tokens_to_string(ch)
                r = score_piece(ch_text)
                tc = len(ch)
                w_pos += r["positive"] * tc
                w_neg += r["negative"] * tc
                total_tok += tc
        if total_tok == 0:
            return None
        return (w_pos - w_neg) / total_tok

    return score_document


# ================================================================
# 1. CentralBankRoBERTa sentiment (primary)
# ================================================================

def load_roberta_scorer():
    model_name = "Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    assert set(id2label.values()) == {"positive", "negative"}
    return build_document_scorer(tokenizer, model, id2label)


def compute_roberta_sentiment(pr_files, qa_files, meeting_ids):
    score_document = load_roberta_scorer()
    rows = []
    for mid in meeting_ids:
        pr_text = pr_files[mid].read_text(encoding="utf-8")
        qa_text = qa_files[mid].read_text(encoding="utf-8")
        combined_text = pr_text + "\n\n" + qa_text  # whole conference = combined text re-scored as one document
        rows.append({
            "Meeting_ID": mid,
            "PR_Sentiment_RoBERTa": score_document(pr_text),
            "QA_Sentiment_RoBERTa": score_document(qa_text),
            "Overall_Sentiment_RoBERTa": score_document(combined_text),
        })
    return pd.DataFrame(rows)


# ================================================================
# 2. Loughran-McDonald uncertainty (primary)
# ================================================================

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def load_uncertainty_dictionary():
    try:
        lm_dictionary = pd.read_csv(LM_DICTIONARY_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        lm_dictionary = pd.read_csv(LM_DICTIONARY_PATH, encoding="latin1")
    return set(
        lm_dictionary.loc[
            pd.to_numeric(lm_dictionary["Uncertainty"], errors="coerce").fillna(0) > 0, "Word"
        ].astype(str).str.upper().str.strip()
    )


def calculate_uncertainty(text, uncertainty_words):
    """Uncertainty = 1000 x (LM uncertainty-word count) / (total word count)."""
    words = [w.upper() for w in WORD_PATTERN.findall(text)]
    if not words:
        return None
    word_counter = Counter(words)
    unc_count = sum(c for w, c in word_counter.items() if w in uncertainty_words)
    return 1000 * unc_count / len(words)


def compute_lm_uncertainty(pr_files, qa_files, meeting_ids):
    uncertainty_words = load_uncertainty_dictionary()
    rows = []
    for mid in meeting_ids:
        pr_text = pr_files[mid].read_text(encoding="utf-8")
        qa_text = qa_files[mid].read_text(encoding="utf-8")
        combined_text = pr_text + "\n\n" + qa_text
        rows.append({
            "Meeting_ID": mid,
            "PR_Uncertainty_LM": calculate_uncertainty(pr_text, uncertainty_words),
            "QA_Uncertainty_LM": calculate_uncertainty(qa_text, uncertainty_words),
            "Overall_Uncertainty_LM": calculate_uncertainty(combined_text, uncertainty_words),
        })
    return pd.DataFrame(rows)


# ================================================================
# 3. FinBERT-FOMC sentiment (robustness)
# ================================================================

def load_finbert_fomc_scorer():
    model_name = "ZiweiChen/FinBERT-FOMC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    assert set(id2label.values()) == {"positive", "negative", "neutral"}
    return build_document_scorer(tokenizer, model, id2label)


def compute_finbert_fomc_sentiment(pr_files, qa_files, meeting_ids):
    score_document = load_finbert_fomc_scorer()
    rows = []
    for mid in meeting_ids:
        pr_text = pr_files[mid].read_text(encoding="utf-8")
        qa_text = qa_files[mid].read_text(encoding="utf-8")
        combined_text = pr_text + "\n\n" + qa_text
        rows.append({
            "Meeting_ID": mid,
            "PR_Sentiment_FinBERT_FOMC": score_document(pr_text),
            "QA_Sentiment_FinBERT_FOMC": score_document(qa_text),
            "Overall_Sentiment_FinBERT_FOMC": score_document(combined_text),
        })
    return pd.DataFrame(rows)


# ================================================================
# 4. Hawkish/dovish stance (robustness, distinct construct)
# Gorodnichenko, Pham & Talavera (2021/2023), "The Voice of Monetary
# Policy", Table 2 dictionary. Prefix matching, sentence-part
# segmentation, negation handling. "highlight" is excluded from
# matching "high"; "highly" and "strongly" are retained.
# ================================================================

panel_A1 = [
    "inflation expectation", "interest rate", "bank rate", "fund rate",
    "price", "economic activity", "inflation", "employment",
]
panel_A2 = [
    "anchor", "cut", "subdue", "declin", "decrease", "reduc", "low",
    "drop", "fall", "fell", "deceler", "slow", "pause", "pausing",
    "stable", "non-accelerating", "downward", "tighten",
]
panel_B1 = [
    "unemployment", "growth", "exchange rate", "productivity",
    "deficit", "demand", "job market", "monetary policy",
]
panel_B2 = [
    "ease", "easing", "rise", "rising", "increase", "expand", "improv",
    "strong", "upward", "raise", "high", "rapid",
]
panel_C = [
    "weren't", "were not", "wasn't", "was not", "did not", "didn't",
    "do not", "don't", "will not", "won't",
]

CONJUNCTIONS = sorted(
    ["and", "but", "while", "however", "although", "because", "so that",
     "even though", "though", "since", "when", "if", "unless", "whereas", "as"],
    key=len, reverse=True,
)

EXCLUDED_MATCHES = {"high": {"highlight", "highlights", "highlighted", "highlighting"}}


def split_into_parts(sentence):
    """Split on semicolons/newlines, then on a comma immediately
    followed by a listed conjunction."""
    coarse_parts = [p.strip() for p in re.split(r"[;\n]+", sentence) if p.strip()]
    conj_pattern = r",\s+(?=(?:" + "|".join(re.escape(c) for c in CONJUNCTIONS) + r")\b)"
    parts = []
    for part in coarse_parts:
        parts.extend([p.strip() for p in re.split(conj_pattern, part, flags=re.IGNORECASE) if p.strip()])
    return parts


def find_matches_in_part(part_text, dictionary):
    """Single-word entries match by prefix (e.g. 'reduction' matches
    'reduc'); multi-word entries match as a substring of the part."""
    words = re.findall(r"[A-Za-z']+", part_text.lower())
    found = []
    single_word_entries = [d for d in dictionary if " " not in d]
    multi_word_entries = [d for d in dictionary if " " in d]
    for w in words:
        for d in single_word_entries:
            if w.startswith(d) and w not in EXCLUDED_MATCHES.get(d, set()):
                found.append(d)
    part_lower = part_text.lower()
    for d in multi_word_entries:
        if d in part_lower:
            found.append(d)
    return found


def check_negation(part_text):
    text_lower = part_text.lower()
    for neg in panel_C:
        if neg in text_lower:
            return neg
    return None


def classify_part(part_text):
    m_A1 = find_matches_in_part(part_text, panel_A1)
    m_A2 = find_matches_in_part(part_text, panel_A2)
    m_B1 = find_matches_in_part(part_text, panel_B1)
    m_B2 = find_matches_in_part(part_text, panel_B2)
    negation = check_negation(part_text)
    results = []
    if m_A1 and m_A2:
        results.append("HAWKISH" if negation else "DOVISH")
    if m_B1 and m_B2:
        results.append("HAWKISH" if negation else "DOVISH")
    if m_A1 and m_B2:
        results.append("DOVISH" if negation else "HAWKISH")
    if m_B1 and m_A2:
        results.append("DOVISH" if negation else "HAWKISH")
    return results


def process_document(text):
    hawkish_count = dovish_count = 0
    for sentence in sent_tokenize(text):
        for part in split_into_parts(sentence):
            for classification in classify_part(part):
                if classification == "HAWKISH":
                    hawkish_count += 1
                else:
                    dovish_count += 1
    return hawkish_count, dovish_count


def hd_sentiment(dovish, hawkish):
    """HD = (Dovish - Hawkish) / (Dovish + Hawkish); positive = relatively
    dovish, negative = relatively hawkish."""
    return None if (dovish + hawkish) == 0 else (dovish - hawkish) / (dovish + hawkish)


def compute_hawkish_dovish_stance(pr_files, qa_files, meeting_ids):
    rows = []
    for mid in meeting_ids:
        pr_text = pr_files[mid].read_text(encoding="utf-8")
        qa_text = qa_files[mid].read_text(encoding="utf-8")
        pr_hawk, pr_dove = process_document(pr_text)
        qa_hawk, qa_dove = process_document(qa_text)
        overall_hawk, overall_dove = pr_hawk + qa_hawk, pr_dove + qa_dove
        rows.append({
            "Meeting_ID": mid,
            "PR_HD_Sentiment": hd_sentiment(pr_dove, pr_hawk),
            "QA_HD_Sentiment": hd_sentiment(qa_dove, qa_hawk),
            "Overall_HD_Sentiment": hd_sentiment(overall_dove, overall_hawk),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pr_files, qa_files, meeting_ids = load_transcript_files()

    roberta_df = compute_roberta_sentiment(pr_files, qa_files, meeting_ids)
    uncertainty_df = compute_lm_uncertainty(pr_files, qa_files, meeting_ids)
    fomc_df = compute_finbert_fomc_sentiment(pr_files, qa_files, meeting_ids)
    hd_df = compute_hawkish_dovish_stance(pr_files, qa_files, meeting_ids)

    with pd.ExcelWriter(TEXT_MEASURES_FILE) as writer:
        roberta_df.to_excel(writer, sheet_name="RoBERTa", index=False)
        uncertainty_df.to_excel(writer, sheet_name="LM_Uncertainty", index=False)
        fomc_df.to_excel(writer, sheet_name="FinBERT_FOMC", index=False)
        hd_df.to_excel(writer, sheet_name="Hawkish_Dovish", index=False)
    print(f"Saved: {TEXT_MEASURES_FILE}")
