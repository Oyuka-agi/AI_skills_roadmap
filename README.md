# AI_skills_roadmap
    2
    3 **Using machine learning to identify professional skill gaps and automate
      personalized career learning roadmaps.**
    4
    5 ---
    6
    7 ##  Project Overview
    8 The **Career Roadmap App** is an interactive, AI-powered platform designed
      to bridge the gap between a candidate's current profile and their target
      career goals. By leveraging a custom Machine Learning pipeline, the tool
      automatically extracts technical competencies from resumes and job
      descriptions to provide a high-precision skill gap analysis.
    9
   10 ##  Key Features
   11 - **ML-Hybrid Skill Extraction**: Combines deterministic taxonomy matching
      with a custom-trained **NER (Named Entity Recognition)** model.
   12 - **Predictive Importance Ranking**: Uses a Linear Regression model
      (`skill_ranker.joblib`) to predict skill relevance based on frequency,
      context (must-have vs. nice-to-have), and document position.
   13 - **Utility-Based Course Optimization**: An intelligent recommendation
      engine that maximizes learning utility based on:
   14     - **Skill Coverage** (45%)
   15     - **English Language Priority** (25%)
   16     - **Course Quality & Difficulty Match** (20%)
   17     - **Time Efficiency** (10%)
   18 - **Interactive Roadmap**: A visual timeline that adapts in real-time to
      user-defined time budgets.
   19
   20 ##  Architecture
   21 - **Backend**: Python, SpaCy (NLP), Scikit-Learn (Ranking Model),
      SkillNER.
   22 - **Frontend**: Streamlit (Interactive Dashboard).
   23 - **Data**: Curated skill taxonomy and a specialized course catalog.
   24
   25 ##  Installation & Setup
   26
   1
   2 2. **Install dependencies**:
     pip install -r requirements.txt
     python -m spacy download en_core_web_sm
   1
   2 3. **Run the Application**:
     streamlit run app.py

   1
   2 ##  Model Logic
   3 Detailed technical documentation for our core models can be found in the
     `docs/` directory:
   4 - [JD Extractor Logic](./docs/jd_extractor_logic.md)
   5 - [Resume Extractor Logic](./docs/resume_extractor_logic.md)
   6 - [Course Optimizer Logic](./docs/course_optimizer_logic.md)
