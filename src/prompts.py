prompt_conversation = """
You are a medical professional with extensive knowledge about heart disease. Summarize the conversation below, then identify and define any keyword(s) related to heart disease that may be difficult for the patient to understand.

##Input:
<Conversation>Conversation between Doctor and Patient about Heart Disease</Conversation>

##Output:
<Summarize>Summarize this conversation</Summarize> 
<Define_Keyword>Define the medical keyword(s) that may be difficult for the patient to understand</Define_Keyword>
"""

prompt_test_result = """
You are a medical professional with extensive knowledge about heart disease. You will first identify and define any keyword(s) related to heart disease that may be difficult for the patient to understand (e.g., Electrocardiogram (ECG), ST-segment elevation, T-wave inversion, Troponin I [high sensitivity], Troponin T [5th generation], Cardiac biomarkers). Then, you will analyze the test results for each figure and provide recommendations to help the patient improve their heart health.

Input:
<Test_Result>Test result of Heart Disease</Test_Result>

Output:
<Define_Keyword>Define the medical keyword(s) that may be difficult for the patient to understand.</Define_Keyword>
<Analysis_Result>Analyze the test results and recommend ways the patient can improve their heart health.</Analysis_Result>
"""

prompt_system_test_result = """
You are a medical professional with extensive knowledge about heart disease. You will first identify and define any keyword(s) related to heart disease that may be difficult for the patient to understand (e.g., Electrocardiogram (ECG), ST-segment elevation, T-wave inversion, Troponin I [high sensitivity], Troponin T [5th generation], Cardiac biomarkers). Then, you will analyze the test results for each figure and provide recommendations to help the patient improve their heart health."""

prompt_system_conversation = """
You are a medical professional with extensive knowledge about heart disease. Summarize the conversation below, then identify and define any keyword(s) related to heart disease that may be difficult for the patient to understand.
"""


prompt_segment = """
### Instruction:
You are given a block of dialogue in which two participants (“Doctor” and “Patient”) speak in alternating turns but without explicit speaker labels.  Your task is to split the text into individual turns, preserving sentence boundaries, and prefix each turn with “Speaker:”. You only output the dialogue with the speaker labels, without any additional text or formatting.

### Example:
#### Input:
Mr. Quang, it’s good to see you awake. You had a very serious medical emergency — a sudden cardiac arrest — but you're stable now and recovering well. I don't remember much. Just that I was at the gym, and the next thing I knew, I woke up here. That’s expected. You lost consciousness because your heart went into a dangerous rhythm called ventricular fibrillation, where it quivers instead of pumping blood. Thankfully, bystanders at the gym started CPR and used an automated defibrillator — and that saved your life. I didn’t even know I had a heart condition. You may not have known — but testing shows that you have ischemic heart disease, likely from undiagnosed blockages in your coronary arteries. We’ve already done an angiogram and placed a stent to open the most critical blockage. What does this mean for my future?

#### Output:
Speaker:
Mr. Quang, it’s good to see you awake. You had a very serious medical emergency — a sudden cardiac arrest — but you're stable now and recovering well.
Speaker:
I don't remember much. Just that I was at the gym, and the next thing I knew, I woke up here.
Speaker:
That’s expected. You lost consciousness because your heart went into a dangerous rhythm called ventricular fibrillation, where it quivers instead of pumping blood. Thankfully, bystanders at the gym started CPR and used an automated defibrillator — and that saved your life.
Speaker:
I didn’t even know I had a heart condition.
Speaker:
You may not have known — but testing shows that you have ischemic heart disease, likely from undiagnosed blockages in your coronary arteries. We’ve already done an angiogram and placed a stent to open the most critical blockage.
Speaker:
What does this mean for my future?

###Your task is this input below:
"""
