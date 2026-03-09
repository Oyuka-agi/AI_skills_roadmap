# ==============================================================================
# skill_model.py 
# ==============================================================================
import pandas as pd
import numpy as np
import re
import ast
import spacy
import pickle
from collections import Counter
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Load NLP model for semantic parsing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class HardSkillExtractionModel:
    """
    An end-to-end Model optimized for Hard Skill extraction (for learning source matching).
    Uses Word2Vec on unlabeled data to discover technical synonyms.
    """
    def __init__(self):
        self.hard_skills_vocab = set()
        self.semantic_model = None
        
        # Aggressive filter to remove generic IT/Business terms and boost Precision
        self.generic_terms = {
            'development', 'deployment', 'software', 'hardware', 'system', 'systems',
            'database', 'databases', 'environment', 'infrastructure', 'platform',
            'application', 'applications', 'technology', 'technologies', 'tool',
            'tools', 'framework', 'frameworks', 'solution', 'solutions', 'process',
            'management', 'support', 'business', 'data', 'analysis', 'design',
            'integration', 'architecture', 'service', 'services', 'project',
            'agile', 'scrum', 'team', 'teams', 'communication', 'skills', 'cycle',
            'product', 'products', 'testing', 'test', 'concept', 'concepts'
        }

    def _clean_text(self, text):
        if not isinstance(text, str): return ""
        # Keep +, # for skills like C++, C#
        text = re.sub(r'[^a-zA-Z0-9\s#\+]', ' ', text.lower())
        return " ".join(text.split())

    def train_semi_supervised(self, gold_df, unlabeled_df, text_col_gold, text_col_unlabeled):
        print("-> [1/3] Extracting strict hard skills from Gold labels...")
        temp_vocab = set()
        for _, row in gold_df.iterrows():
            try:
                skills = ast.literal_eval(row['job_skill_set']) if isinstance(row['job_skill_set'], str) else row['job_skill_set']
                for sk in skills:
                    sk_clean = self._clean_text(sk)
                    # Filter out short words, generic terms, and pure numbers
                    if (len(sk_clean) > 2 and 
                        not sk_clean.isdigit() and 
                        sk_clean not in self.generic_terms):
                        temp_vocab.add(sk_clean)
            except:
                continue
        
        print("-> [2/3] Training Word2Vec on Unlabeled Data for contextual awareness...")
        corpus_texts = unlabeled_df[text_col_unlabeled].dropna().sample(n=min(30000, len(unlabeled_df)), random_state=42)
        sentences = [self._clean_text(text).split() for text in corpus_texts]
        self.semantic_model = Word2Vec(sentences, vector_size=50, window=5, min_count=5, workers=4)
        
        print("-> [3/3] Expanding vocab via Semi-supervised learning...")
        self.hard_skills_vocab.update(temp_vocab)
        expanded_skills = set()
        for skill in temp_vocab:
            if len(skill.split()) == 1 and skill in self.semantic_model.wv.key_to_index:
                similar_words = self.semantic_model.wv.most_similar(skill, topn=2)
                for word, score in similar_words:
                    if score > 0.75 and word not in self.generic_terms and len(word) > 2:
                        expanded_skills.add(word)
                        
        self.hard_skills_vocab.update(expanded_skills)
        print(f"=== Training Completed. Dictionary contains {len(self.hard_skills_vocab)} verified hard skills. ===")

    def predict(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return []

        doc = nlp(text)
        text_lower = text.lower()
        extracted_candidates = set()
        
        for skill in self.hard_skills_vocab:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                extracted_candidates.add(skill)
                
        words_in_text = text_lower.split()
        word_counts = Counter(words_in_text)
        total_words = len(words_in_text)
        results = []
        
        for skill in extracted_candidates:
            skill_tokens = skill.split()
            freq = word_counts[skill_tokens[0]] if skill_tokens[0] in word_counts else 1
            
            base_score = 5.0
            freq_boost = min((freq / max(total_words, 1)) * 50, 3.0) 
            specificity_boost = 1.5 if len(skill_tokens) > 1 else 0.0
            relevance = min(round(base_score + freq_boost + specificity_boost, 1), 9.9)
            
            context_sentences = [sent.text.lower() for sent in doc.sents if skill in sent.text.lower()]
            context = " ".join(context_sentences)
            
            difficulty = "Intermediate" 
            if any(w in context for w in ['senior', 'expert', 'lead', 'architecture', 'extensive', 'advanced', 'deep']):
                difficulty = "Advanced"
            elif any(w in context for w in ['junior', 'entry', 'basic', 'familiar', 'understanding', 'intern', 'exposure']):
                difficulty = "Entry-Level"
                
            results.append({
                "skill": skill.title(),
                "relevance_score": relevance,
                "difficulty_level": difficulty
            })
            
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)

    def save_model(self, filepath="hard_skill_model.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump({'vocab': self.hard_skills_vocab, 'w2v': self.semantic_model}, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="hard_skill_model.pkl"):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.hard_skills_vocab = data['vocab']
            self.semantic_model = data['w2v']
        print(f"Model loaded from {filepath}")