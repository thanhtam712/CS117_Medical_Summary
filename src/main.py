import os
import sys
import argparse
from loguru import logger
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from natsort import natsorted

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core import ChatPromptTemplate

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.prompts import prompt_conversation, prompt_test_result, prompt_system_conversation, prompt_system_test_result
from src.preprocess.audio_preprocessing import AudioProcessor
from src.preprocess.pdf_preprocessing import PDFProcessor

load_dotenv()

llm_conversation = OpenAI(
    "gpt-4.1-mini",
    api_key=os.getenv("API_OPENAI_KEY"),
    system_prompt=prompt_system_conversation,
)

llm_test_result = OpenAI(
    "gpt-4.1-mini",
    api_key=os.getenv("API_OPENAI_KEY"),
    system_prompt=prompt_system_test_result,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with GPT about heart disease.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file containing heart disease information.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file containing heart disease information.")
    return parser.parse_args()

def chat_gpt(input: str, type: str = Literal["conversation", "test_result"]) -> str:
    if type == "conversation":
        prompt_gpt = prompt_conversation
        llm = llm_conversation
        prompt_system = prompt_system_conversation
    elif type == "test_result":
        prompt_gpt = prompt_test_result
        llm = llm_test_result
        prompt_system = prompt_system_test_result
    else:
        raise ValueError("Type must be either 'conversation' or 'test_result'.")

    prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role="system",
                content=prompt_system,
            ),
            ChatMessage(
                role="user",
                content=str(input) + "\n" + prompt_gpt,
            ),
        ]
    )

    messages = prompt.format_messages()
    response = llm.chat(messages)
    response = response.message.content

    return response

def preprocess_pdf(pdf_path: str, pdf_process: PDFProcessor) -> str:
    texts_parsed = pdf_process.convert(str(pdf_path))
    texts_structured = pdf_process.restructure(texts_parsed)
    
    return texts_structured

def preprocess_audio(audio_path: str, audio_process: AudioProcessor) -> str:
    texts_predict = audio_process.speech_recognition(str(audio_path))
    roles_predict = audio_process.classify_role(texts_predict)
    
    text_copy = texts_predict.copy()
    roles_copy = roles_predict.copy()
    texts_predict = []
    roles_predict = []
    for idx, text in enumerate(text_copy):
        if idx >= 1 and roles_copy[idx] == roles_copy[idx - 1]:
            texts_predict[-1] += " " + text
        else:
            texts_predict.append(text)
            roles_predict.append(roles_copy[idx])
            
    conversation = "\n".join(
        f"{role.capitalize()}: {text.strip()}" for role, text in zip(roles_predict, texts_predict)
    )
    
    return conversation

def load_model():
    return PDFProcessor(), AudioProcessor()
   
def main(pdf_process: PDFProcessor, audio_process: AudioProcessor):
    args = parse_args()
    pdf_path = args.pdf
    audio_path = args.audio

    logger.info("Preprocessing PDF and audio files...")
    conversation_preprocessed = preprocess_audio(audio_path, audio_process)
    test_result_preprocessed = preprocess_pdf(pdf_path, pdf_process)
    
    logger.info("Requesting ChatGPT for test result and conversation summaries...")
    final_test_result = chat_gpt(test_result_preprocessed, type="test_result")
    final_conversation = chat_gpt(conversation_preprocessed, type="conversation")
    
    logger.info("Combining final summaries...")
    final_summarize_combine = final_test_result + "\n" + final_conversation
    
    output_file = "final_summary.txt"
    with open(output_file, "w") as f:
        f.write(final_summarize_combine)

if __name__ == "__main__":
    logger.info("Loading PDFProcessor and AudioProcessor...")
    pdf_process, audio_process = load_model()

    main(pdf_process, audio_process)
