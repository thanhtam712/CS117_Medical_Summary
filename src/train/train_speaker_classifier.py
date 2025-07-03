import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils import compute_class_weight

RANDOM_STATE = 42

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    """Generate handcrafted features known to help doctor vs patient split."""

    def __init__(self):
        self.doctor_keywords = [
            "examine", "prescribe", "diagnose", "recommend", "suggest", "advise",
            "treatment", "therapy", "procedure", "surgery", "operation",
            "medicine", "medication", "drug", "dose", "dosage", "tablet",
            "pill", "hospital", "clinic", "medical", "health", "condition",
            "disease", "based on", "according to", "in my opinion",
            "let me", "i need to", "we will", "we should", "i recommend",
            "you should", "you need", "test", "result", "scan", "x-ray",
            "blood work", "lab", "report", "normal", "abnormal", "positive",
            "negative", "doctor", "physician", "nurse", "specialist",
            "consultant",
        ]

        self.patient_keywords = [
            "feel", "feeling", "hurt", "hurts", "pain", "painful", "ache",
            "aching", "sick", "ill", "unwell", "tired", "weak", "dizzy",
            "nauseous", "worry", "worried", "afraid", "scared", "anxious",
            "concerned", "upset", "frustrated", "confused", "help me",
            "can you", "could you", "will i", "am i", "is it", "what should i",
            "how long", "when will", "why do i", "i have", "i feel", "i am",
            "i was", "i've been", "my pain", "my condition", "it hurts when",
            "i can't", "i couldn't", "thank you", "thanks", "please", "sorry",
            "excuse me",
        ]

        self.symptom_keywords = [
            "headache", "fever", "cough", "cold", "flu", "chest pain",
            "back pain", "stomach", "nausea", "vomiting", "diarrhea",
            "shortness of breath", "breathing", "heart", "blood pressure",
            "diabetes", "infection", "allergy", "rash", "swelling",
        ]

        self.emotion_words = [
            "worried", "scared", "concerned", "anxious", "upset",
            "frustrated",
        ]

    def fit(self, X, y=None): 
        return self

    def transform(self, texts: List[str]):
        feats: List[List[float]] = []
        for text in texts:
            t = text.lower().strip()
            words = t.split()

            doc_cnt = sum(1 for kw in self.doctor_keywords if kw in t)
            pat_cnt = sum(1 for kw in self.patient_keywords if kw in t)
            sym_cnt = sum(1 for kw in self.symptom_keywords if kw in t)

            is_q = 1 if "?" in t else 0
            is_exc = 1 if "!" in t else 0

            first_p = len(re.findall(r"\b(i|me|my|myself|i'm|i've|i'll)\b", t))
            second_p = len(re.findall(r"\b(you|your|yourself|you're|you've|you'll)\b", t))

            n_words = len(words)
            avg_w_len = np.mean([len(w) for w in words]) if words else 0
            medical_density = (doc_cnt + sym_cnt) / max(n_words, 1)
            comma_cnt = t.count(",")
            sent_complex = comma_cnt / max(n_words, 1)

            emo_cnt = sum(1 for kw in self.emotion_words if kw in t)
            prof_cnt = len(re.findall(r"\b(i recommend|you should|we need to|based on|in my experience|let me)\b", t))
            patient_q_cnt = len(re.findall(r"\b(what should i|how long|when will|am i|will i|can you help)\b", t))

            feats.append(
                [
                    doc_cnt,
                    pat_cnt,
                    sym_cnt,
                    is_q,
                    is_exc,
                    first_p,
                    second_p,
                    n_words,
                    avg_w_len,
                    medical_density,
                    sent_complex,
                    emo_cnt,
                    prof_cnt,
                    patient_q_cnt,
                ]
            )
        return csr_matrix(feats, dtype=np.float32)


def preprocess_text(txt: str) -> str:
    """Basic cleaning; can be extended later."""
    txt = re.sub(r"[^\w\s\?\!\.]", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def load_txt_dataset(folder: str) -> Tuple[List[str], List[str]]:
    """Assumes each .txt file alternates role label and utterance line by line."""
    texts, labels = [], []

    for file in Path(folder).glob("*.txt"):
        with open(file, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        # Expect pattern: role, content, role, content ...
        for i in range(0, len(lines) - 1, 2):
            role, sentence = lines[i].lower(), preprocess_text(lines[i + 1])
            if "patient" in role:
                labels.append("patient")
            else:
                labels.append("doctor")
            texts.append(sentence)

    logger.info("Loaded %d samples from %s", len(texts), folder)
    return texts, labels

class SpeakerClassifierTrainer:
    """Train and persist the best model for doctor/patient sentence tagging."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words="english",
            lowercase=True,
            min_df=2,
            max_df=0.9,
        )

        self.custom_feat = CustomFeatureExtractor()

        self.pipeline = Pipeline(
            steps=[
                (
                    "union",
                    FeatureUnion(
                        transformer_list=[
                            ("tfidf", self.vectorizer),
                            ("custom", self.custom_feat),
                        ],
                        n_jobs=-1,
                    ),
                ),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        self.best_estimator_ = None

    def train(self, texts: List[str], labels: List[str]):
        X, y = texts, labels

        param_grid = {
            "clf__C": [0.5, 1.0, 2.0, 4.0],
            "clf__penalty": ["l2"],
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        grid = GridSearchCV(
            self.pipeline,
            param_grid=param_grid,
            cv=skf,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X, y)
        self.best_estimator_ = grid.best_estimator_

        logger.info("Best params: ", grid.best_params_)
        logger.info("Best CV F1: ", grid.best_score_)
        return grid.best_score_

    @staticmethod
    def _pretty_print_cm(cm: np.ndarray, labels: List[str]):
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        logger.info("Confusion Matrix:\n%s", df_cm.to_string())

    def evaluate(self, texts: List[str], labels: List[str]):
        if not self.best_estimator_:
            raise RuntimeError("Model has not been trained yet!")

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
        )

        self.best_estimator_.fit(X_train, y_train)
        y_pred = self.best_estimator_.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted")
        logger.info("Test F1: ", f1)

        logger.info("Classification Report: ", classification_report(y_test, y_pred))
        self._pretty_print_cm(confusion_matrix(y_test, y_pred), self.best_estimator_.classes_)

        return f1

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.best_estimator_, os.path.join(out_dir, "speaker_clf.joblib"))
        logger.info("Saved model to ", out_dir)
        
def parse_args():
    ap = argparse.ArgumentParser(description="Doctor vs patient sentence classifier trainer")
    ap.add_argument("--data_dir", required=True, help="Directory containing .txt transcript files")
    ap.add_argument("--output_dir", required=True, help="Directory to save trained model and artifacts")
    return ap.parse_args()


def main(data_dir: str, output_dir: str):
    texts, labels = load_txt_dataset(data_dir)

    trainer = SpeakerClassifierTrainer()
    logger.info("Starting hyperparameter search and training ...")
    trainer.train(texts, labels)

    logger.info("Evaluating on hold‑out set ...")
    trainer.evaluate(texts, labels)

    trainer.save(output_dir)
    logger.info("Training pipeline complete.")
    
    texts = ["Good afternoon, Mr. Glover. I’m Dr. Arias. Thanks for coming in today. How are you feeling?", "Afternoon, Doctor. A bit unsettled, to be honest. The chest pain the other day really scared me.", "I can imagine. Sudden chest discomfort is always alarming. Tell me more—what exactly did you feel?", "It started out of nowhere. I was just sitting down when I felt this pressure in my chest. Not sharp, but heavy—like something was squeezing. It lasted about half an hour.", "Did you have any other symptoms while it was happening? Nausea? Sweating? Feeling faint?", "Yeah, actually. I got nauseous right after the pain started. Then I began sweating a lot, and at one point, I thought I was going to pass out."]
    
    labels = ["doctor", "patient", "doctor", "patient", "doctor", "patient"]
    
    for text, label in zip(texts, labels):
        processed_text = preprocess_text(text)
        prediction = trainer.best_estimator_.predict([processed_text])[0]
        print(f"Text: '{text}' | True: {label} | Predicted: {prediction}")

if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.output_dir)
