"""
resume_extractor.py

Resume skill extraction module using SkillNER + spaCy.

Purpose:
- Clean raw resume text
- Load SkillNER only once
- Extract skills from one resume or a batch of resumes
- Be directly importable into a dashboard / Streamlit app

Requirements:
    pip install skillNer spacy pandas
    python -m spacy download en_core_web_lg

If en_core_web_lg is unavailable, you can switch to en_core_web_md.
"""

import re
import warnings
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# 1. TEXT CLEANING
# =========================================================
def clean_resume_text(text: Optional[Union[str, Any]]) -> str:
    """
    Clean raw resume text before passing into SkillNER.

    Parameters
    ----------
    text : str or None
        Raw resume text.

    Returns
    -------
    str
        Lowercased and cleaned resume text.
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove HTML tags if they exist
    text = re.sub(r"<.*?>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# 2. LOAD SPACY + SKILLNER ONCE
# =========================================================
@lru_cache(maxsize=1)
def load_skillner() -> SkillExtractor:
    """
    Load spaCy model and SkillNER only once.

    Returns
    -------
    SkillExtractor
        Initialized SkillNER extractor.

    Notes
    -----
    This is intentionally not serialized with joblib.
    We load the pretrained pipeline directly and cache it in memory.
    """
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        # Fallback if large model is not installed
        nlp = spacy.load("en_core_web_md")

    extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return extractor


# =========================================================
# 3. LOW-LEVEL SKILLNER EXTRACTION
# =========================================================
def _parse_skillner_matches(matches_obj: Any) -> List[str]:
    """
    Parse SkillNER output safely across different versions.

    Parameters
    ----------
    matches_obj : Any
        Usually a list or dict returned by SkillNER.

    Returns
    -------
    List[str]
        Extracted normalized skill strings.
    """
    extracted = []

    if isinstance(matches_obj, dict):
        iterable = matches_obj.values()
    elif isinstance(matches_obj, list):
        iterable = matches_obj
    else:
        iterable = []

    for match in iterable:
        if isinstance(match, dict):
            skill = (
                match.get("doc_node_value")
                or match.get("doc_node_name")
                or match.get("skill_id")
            )
            if skill:
                extracted.append(str(skill).lower())

    return extracted


def extract_skills_skillner(text: str, include_ngram_scored: bool = False) -> List[str]:
    """
    Extract skills from cleaned resume text using SkillNER.

    Parameters
    ----------
    text : str
        Cleaned resume text.
    include_ngram_scored : bool, default=False
        Whether to include ngram_scored results.
        False is recommended because it is faster and usually less noisy.

    Returns
    -------
    List[str]
        Deduplicated list of extracted skills.
    """
    if not text:
        return []

    extractor = load_skillner()

    try:
        annotations = extractor.annotate(text)
        results = annotations.get("results", {})

        extracted = []

        # Safer / faster baseline
        full_matches = results.get("full_matches", [])
        extracted.extend(_parse_skillner_matches(full_matches))

        # Optional: include approximate matches
        if include_ngram_scored:
            ngram_scored = results.get("ngram_scored", [])
            extracted.extend(_parse_skillner_matches(ngram_scored))

        # Deduplicate while preserving order
        extracted = list(dict.fromkeys(extracted))
        return extracted

    except Exception:
        return []


# =========================================================
# 4. PUBLIC INFERENCE FUNCTIONS
# =========================================================
def extract_resume_skills(
    resume_text: Optional[Union[str, Any]],
    include_ngram_scored: bool = False
) -> List[str]:
    """
    Full pipeline for one resume:
    raw text -> clean -> SkillNER -> skills

    Parameters
    ----------
    resume_text : str or None
        Raw resume text.
    include_ngram_scored : bool, default=False
        Whether to include approximate n-gram matches.

    Returns
    -------
    List[str]
        Extracted skills.
    """
    cleaned = clean_resume_text(resume_text)
    return extract_skills_skillner(
        text=cleaned,
        include_ngram_scored=include_ngram_scored
    )


def extract_resume_skills_batch(
    resume_texts: List[Optional[Union[str, Any]]],
    include_ngram_scored: bool = False
) -> List[List[str]]:
    """
    Extract skills for a batch of resumes.

    Parameters
    ----------
    resume_texts : list
        List of raw resume texts.
    include_ngram_scored : bool, default=False
        Whether to include approximate n-gram matches.

    Returns
    -------
    List[List[str]]
        Extracted skills for each resume.
    """
    return [
        extract_resume_skills(text, include_ngram_scored=include_ngram_scored)
        for text in resume_texts
    ]


def extract_resume_record(
    resume_id: Any,
    resume_text: Optional[Union[str, Any]],
    include_ngram_scored: bool = False
) -> Dict[str, Any]:
    """
    Extract skills and return a structured record for one resume.

    Parameters
    ----------
    resume_id : Any
        Resume identifier.
    resume_text : str or None
        Raw resume text.
    include_ngram_scored : bool, default=False
        Whether to include approximate n-gram matches.

    Returns
    -------
    Dict[str, Any]
        Structured result dictionary.
    """
    cleaned = clean_resume_text(resume_text)
    skills = extract_skills_skillner(cleaned, include_ngram_scored=include_ngram_scored)

    return {
        "ID": resume_id,
        "resume_clean": cleaned,
        "skills_skillner": skills
    }


def extract_resume_dataframe(
    df: pd.DataFrame,
    text_col: str = "Resume_str",
    id_col: str = "ID",
    include_ngram_scored: bool = False
) -> pd.DataFrame:
    """
    Apply resume extraction to a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing resume text.
    text_col : str, default='Resume_str'
        Column containing raw resume text.
    id_col : str, default='ID'
        ID column name.
    include_ngram_scored : bool, default=False
        Whether to include approximate n-gram matches.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with:
        - resume_clean
        - skills_skillner
    """
    out = df.copy()

    if text_col not in out.columns:
        raise ValueError(f"Column '{text_col}' not found in dataframe.")

    out["resume_clean"] = out[text_col].apply(clean_resume_text)
    out["skills_skillner"] = out["resume_clean"].apply(
        lambda x: extract_skills_skillner(x, include_ngram_scored=include_ngram_scored)
    )

    return out


# =========================================================
# 5. OPTIONAL HELPER FOR DASHBOARD
# =========================================================
def resume_to_skill_payload(
    resume_text: Optional[Union[str, Any]],
    include_ngram_scored: bool = False
) -> Dict[str, Any]:
    """
    Dashboard-friendly output for one resume.

    Returns
    -------
    Dict[str, Any]
        Example:
        {
            "resume_clean": "...",
            "skills": [...],
            "num_skills": 12
        }
    """
    cleaned = clean_resume_text(resume_text)
    skills = extract_skills_skillner(cleaned, include_ngram_scored=include_ngram_scored)

    return {
        "resume_clean": cleaned,
        "skills": skills,
        "num_skills": len(skills)
    }


# =========================================================
# 6. TEST BLOCK
# =========================================================
if __name__ == "__main__":
    sample_resume = """
    HR Manager with experience in recruiting, onboarding, payroll,
    employee relations, compliance, training and development, and budgeting.
    """

    result = resume_to_skill_payload(sample_resume, include_ngram_scored=False)

    print("=" * 60)
    print("SkillNER Resume Extraction Test")
    print("=" * 60)
    print("Cleaned Resume:")
    print(result["resume_clean"])
    print("\nExtracted Skills:")
    print(result["skills"])
    print("\nNumber of Skills:")
    print(result["num_skills"])