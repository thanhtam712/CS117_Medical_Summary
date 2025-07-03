import re
import sys
import argparse
from pathlib import Path
from loguru import logger
from nltk.translate.bleu_score import corpus_bleu

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocess.audio_preprocessing import AudioProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Audio Processing Performance")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/Conversation/Audio",
        help="Path to the directory containing audio files for evaluation.",
    )
    parser.add_argument(
        "--transcript_dir",
        type=str,
        default="data/Conversation/Transcript",
        help="Path to the directory containing transcript files for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Conversation/output_eval",
        help="Path to the directory where evaluation results will be saved.",
    )
    return parser.parse_args()

def preprocess(text):
    text = text.replace("’", "'").replace("—", " ").replace("–", " ")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r'^(doctor|patient):\s*', '', line, flags=re.IGNORECASE)
        if line:
            lines.append(line)
    text = ' '.join(lines)
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_wer(reference, hypothesis):
	ref_words = reference.split("\n")
	hyp_words = hypothesis.split("\n")

	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if preprocess(ref) != preprocess(hyp))
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)

	total_words = len(ref_words)

	wer = (substitutions + deletions + insertions) / total_words
	return wer

def evaluate(audio_path: str | Path, transcript_path: str | Path, output_file: str | Path, audio_process: AudioProcessor):
    """
    Evaluate the audio processing performance by comparing system outputs with ground truth transcripts.
    """
    logger.info("Starting evaluation of audio processing performance...")
    
    if isinstance(audio_path, str):
        audio_path = Path(audio_path)
    if isinstance(transcript_path, str):
        transcript_path = Path(transcript_path)    
    
    texts_predict = audio_process.speech_recognition(str(audio_path))
    roles_predict = audio_process.classify_role(texts_predict)
    
    transcript_data = open(transcript_path, "r", encoding="utf-8")
    texts_gt = ""
    for line in transcript_data.readlines():
        line = line.strip()
        if not line:
            continue
        
        if "Doctor" in line or "Patient" in line:
            texts_gt += line.strip() + "\n"
        else:
            logger.warning(f"Unknown role in transcript: {line}. Skipping.")
            continue
    
    texts_pre = ""
    texts_predict, roles_predict = [], []
    text_copy = texts_predict.copy()
    roles_copy = roles_predict.copy()
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, text in enumerate(text_copy):
            if idx >= 1 and roles_copy[idx] == roles_copy[idx - 1]:
                texts_predict[-1] += " " + text
            else:
                texts_predict.append(text)
                roles_predict.append(roles_copy[idx])
            
        for text, role in zip(texts_predict, roles_predict):
            f.write(f"{role.capitalize()}: {text}\n")
            texts_pre += f"{role.capitalize()}: {text}\n"
    
    wer_score = calculate_wer(texts_gt.strip(), texts_pre.strip())
    print(f"Word Error Rate (WER): {audio_path.stem} {wer_score:.4f}")
    return wer_score
        
if __name__ == "__main__":
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    transcript_dir = Path(args.transcript_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_process = AudioProcessor(whisper_model="turbo", classify_model="models/speaker_clf.joblib")
    
    wer_scores = []
    for audio_path in audio_dir.iterdir():
        file_name = audio_path.stem
        transcript_path = transcript_dir / f"{file_name}.txt"
        output_file = output_dir / f"{file_name}.txt"
        wer_score = evaluate(audio_path, transcript_path, output_file, audio_process)
        wer_scores.append(wer_score)
    
    average_wer = sum(wer_scores) / len(wer_scores)
    print(f"Average Word Error Rate (WER) across all files: {average_wer:.4f}")
