import re
import sys
import joblib
import __main__
import whisper
from pathlib import Path
from typing import List
from loguru import logger
from vllm import LLM, SamplingParams

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.train.train_speaker_classifier import preprocess_text, CustomFeatureExtractor as abc
from src.prompts import prompt_segment

# print(__name__)
# __main__.CustomFeatureExtractor = CustomFeatureExtractor
CustomFeatureExtractor = abc

class AudioProcessor:
    def __init__(self, whisper_model: str = "turbo", classify_model: str = "models/speaker_clf.joblib"):
        logger.info("Initialize AudioProcessor...")
        self.whisper_model = self.load_model_whisper(whisper_model)
        self.classify_model = self.load_model_classify(classify_model)
        self.segment_model, self.sampling_params = self.load_model_segment("data_llama_factory/segment_export")
        
    def load_model_whisper(self, whisper_model):
        model_whisper = whisper.load_model(whisper_model)
        return model_whisper

    def load_model_classify(self, classify_model):
        model_classify = joblib.load(classify_model)
        return model_classify
    
    def load_model_segment(self, segment_model):
        model_segment = LLM(
            model=segment_model,
            trust_remote_code=True,
            gpu_memory_utilization=0.5,
        )
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
        return model_segment, sampling_params
        
    def speech_recognition(self, audio_path: str | Path):
        results_whisper = self.whisper_model.transcribe(audio_path, language="en")["text"]
        
        logger.info("Transcription completed.")
        input = {"prompt": prompt_segment + "\n" + results_whisper}
        output_segment = self.segment_model.generate([input], self.sampling_params)
        
        segments = output_segment[0].outputs[0].text.strip()
        
        print(segments)
        print("=======================")
        texts_output = segments.split("Speaker:\n")[1:]
        texts = []
        for text in texts_output:
            if "###" in text:
                texts.append(text.split("###")[0].strip())
                break
            else:
                texts.append(text.strip())
        
        print(texts)
                
        return texts
    
    def classify_role(self, texts: List[str]):
        roles = []
        for text in texts:
            text = preprocess_text(text)
            role = self.classify_model.predict([text])[0]
            roles.append(str(role).lower())
        return roles
