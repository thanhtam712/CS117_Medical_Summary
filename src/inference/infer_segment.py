import os
import glob
import random
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="data_llama_factory/segment_export",
    trust_remote_code=True,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)

instruction1 = "Hi"

inputs = {"prompt": instruction1, "multi_modal_data": {"image": image}}


outputs = llm.generate([inputs], sampling_params)

result = outputs[0].outputs[0].text
print(result)
