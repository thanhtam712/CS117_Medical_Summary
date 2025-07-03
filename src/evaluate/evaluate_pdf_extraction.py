import os
import re
import sys
import argparse
from loguru import logger
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocess.pdf_preprocessing import PDFProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PDF extraction performance by comparing with ground truth."
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default="data/Test_Result/Parse_Text",
        help="Path to the directory containing ground truth .txt files.",
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/Test_Result/PDF_Image",
        help="Path to the directory containing PDF files to evaluate.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/Test_Result/output_eval",
    )
    return parser.parse_args()



def calculate_wer(reference, hypothesis):
	ref_words = reference.split()
	hyp_words = hypothesis.split()

	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)

	total_words = len(ref_words)

	wer = (substitutions + deletions + insertions) / total_words
	return wer


def evaluate(pdf_path: str | Path, transcript_path: str | Path, output_path: str | Path, pdf_process: PDFProcessor):
    """
    Evaluate the audio processing performance by comparing system outputs with ground truth transcripts.
    """
    logger.info("Starting evaluation of audio processing performance...")
    
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    if isinstance(transcript_path, str):
        transcript_path = Path(transcript_path)   
    if isinstance(output_path, str):
        output_path = Path(output_path) 
    
    texts_parsed = pdf_process.convert(str(pdf_path))
    texts_structured = pdf_process.restructure(texts_parsed)
    
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(texts_structured)
    
    transcript_data = open(transcript_path, "r", encoding="utf-8")
    texts_gt = ""
    for line in transcript_data.readlines():
        line = line.strip()
        if not line:
            continue
        texts_gt += line.strip() + "\n"
    transcript_data.close()
    
    wer_score = calculate_wer(texts_gt, texts_structured)
    print(f"Word Error Rate (WER) for {pdf_path.name}: {wer_score:.4f}")
    return wer_score    
    
if __name__ == "__main__":
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir = Path(args.ground_truth_dir)
    
    pdf_process = PDFProcessor()
    
    wer_scores = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        pdf_name = pdf_file.stem
        transcript_file = ground_truth_dir / f"{pdf_name}.txt"
        output_file = output_dir / f"{pdf_name}.txt"
        wer_score = evaluate(pdf_file, transcript_file, output_file, pdf_process)
        wer_scores.append(wer_score)
    
    average_wer = sum(wer_scores) / len(wer_scores)
    print(f"Average Word Error Rate (WER) across all files: {average_wer:.4f}")
