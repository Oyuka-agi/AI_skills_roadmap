# AI_skills_roadmap

An end-to-end career upskilling system that extracts skills from resumes and job descriptions, identifies skill gaps, and generates a personalized learning roadmap under time constraints.

---

## Project Overview

This project helps users answer a practical question:

> What skills am I missing for a target job, and what should I learn first given limited time?

The system integrates:

- Resume skill extraction
- Job description skill extraction
- Gap analysis
- Course recommendation under a time budget
- Interactive dashboard deployment

---

## System Pipeline

```text
Resume PDF / Text
        ↓
Resume Skill Extractor
        ↓
Resume Skill Set

Target Job Description
        ↓
JD Skill Extraction Model
        ↓
Ranked JD Skill Set (relevance + difficulty + priority)

Resume Skills + JD Skills
        ↓
Gap Analysis
        ↓
Missing Skills

Missing Skills + Course Dataset + Time Budget
        ↓
Optimization Engine
        ↓
Personalized Learning Roadmap
```

## 🔍 Models Used

### 1. Resume Skill Extraction
- **Model:** Pre-trained SkillNER (spaCy-based)
- **Type:** Named Entity Recognition (NER)
- **Purpose:** Extract structured skills from resume text
- **Characteristics:**
  - High precision on known skills
  - Standardized output format
  - No training required (inference-only)

---

### 2. Job Description Skill Extraction
- **Model Type:** Hybrid (rule-based + machine learning)

**Components:**
- **Alias Dictionary Matcher**
  - High precision for known skills

- **ML-based NER**
  - Detects unseen or out-of-vocabulary skills

**Post-processing:**
- Merge and deduplicate extracted skills
- Compute:
  - **Relevance score**
  - **Difficulty score**
  - **Priority ranking**

---

### 3. Optimization Model
