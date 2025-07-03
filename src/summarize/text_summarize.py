import os
from openai import OpenAI
from typing import Dict, Optional
import json


class TextSummarizer:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the text summarizer with OpenAI API"""
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def _get_conversation_prompt(self):
        """Get prompt for conversation summarization"""
        return """
        You are a medical assistant. Analyze the following doctor-patient conversation and provide a structured summary.

        Conversation Text:
        {conversation_text}

        Please provide a summary in the following format:

        **Symptoms/Concerns Raised by Patient:**
        [List the main symptoms and concerns the patient mentioned]

        **Doctor's Diagnosis or Insight:**
        [Summarize the doctor's assessment, diagnosis, or medical insights]

        **Treatment Plan / Next Steps:**
        [Detail any treatments prescribed or next steps recommended]

        Use clear, non-technical language and explain any medical terms that appear.
        """

    def _get_test_analysis_prompt(self):
        """Get prompt for test results analysis"""
        return """
        You are a medical assistant specializing in cardiovascular health. Analyze the following test results.

        Test Results:
        {test_results}

        Please provide analysis in the following format:

        **Highlighted Indicators:**
        [List key test values like LDL, ECG findings, blood pressure, etc.]

        **Abnormal Values:**
        [Identify any values outside normal ranges with explanations]

        **Implications:**
        [Explain what these results mean for cardiovascular health]

        **Actionable Warnings:**
        [Any urgent concerns or recommendations based on the results]

        Define medical terms clearly and explain how each abnormal value relates to heart health.
        """

    def _get_final_summary_prompt(self):
        """Get prompt for final medical summary"""
        return """
        Create a comprehensive medical summary combining the conversation and test analysis.

        Conversation Summary:
        {conversation_summary}

        Test Analysis:
        {test_analysis}

        Please provide a final summary (150-250 words) in this format:

        **Patient's Heart Health Status:**
        [Overall cardiovascular health assessment]

        **Doctor's Summary and Actions:**
        [Key points from doctor consultation and prescribed actions]

        **Test-Based Health Insights:**
        [Important findings from test results]

        **Next Recommended Steps:**
        [Clear action items for the patient]

        Use accessible language for non-medical users while maintaining accuracy.
        """

    def summarize_conversation(self, conversation_text: str) -> Optional[str]:
        """Summarize doctor-patient conversation"""
        if not conversation_text:
            return None

        prompt = self._get_conversation_prompt().format(
            conversation_text=conversation_text
        )

        return self._call_openai_api(prompt, "conversation summary")

    def analyze_test_results(self, test_results: Dict) -> Optional[str]:
        """Analyze medical test results"""
        if not test_results:
            return None

        # Format test results for analysis
        formatted_results = self._format_test_results(test_results)

        prompt = self._get_test_analysis_prompt().format(test_results=formatted_results)

        return self._call_openai_api(prompt, "test analysis")

    def create_final_summary(
        self, conversation_summary: str, test_analysis: str
    ) -> Optional[str]:
        """Create final medical summary"""
        if not conversation_summary or not test_analysis:
            return None

        prompt = self._get_final_summary_prompt().format(
            conversation_summary=conversation_summary, test_analysis=test_analysis
        )

        return self._call_openai_api(prompt, "final summary")

    def _format_test_results(self, test_results: Dict) -> str:
        """Format test results for analysis"""
        if isinstance(test_results, dict) and "structured_data" in test_results:
            data = test_results["structured_data"]

            formatted = []
            formatted.append("PATIENT INFORMATION:")
            if "patient_info" in data:
                for key, value in data["patient_info"].items():
                    formatted.append(f"- {key.title()}: {value}")

            formatted.append("\nTEST RESULTS:")
            if "test_results" in data:
                for result in data["test_results"]:
                    status = "NORMAL" if result.get("normal", True) else "ABNORMAL"
                    formatted.append(
                        f"- {result['test']}: {result['value']} {result.get('unit', '')} [{status}]"
                    )

            formatted.append("\nABNORMAL VALUES:")
            if "abnormal_values" in data:
                for result in data["abnormal_values"]:
                    formatted.append(
                        f"- {result['test']}: {result['value']} {result.get('unit', '')} [ABNORMAL]"
                    )

            if "raw_text" in test_results:
                formatted.append(f"\nRAW TEXT:\n{test_results['raw_text'][:1000]}...")

            return "\n".join(formatted)
        else:
            return str(test_results)

    def _call_openai_api(self, prompt: str, task_name: str) -> Optional[str]:
        """Call OpenAI API with error handling"""
        try:
            print(f"Đang tạo {task_name} bằng LLM...")

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical assistant AI specializing in cardiovascular health.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )

            result = response.choices[0].message.content
            print(f"Tạo {task_name} thành công!")
            return result

        except Exception as e:
            print(f"Lỗi khi gọi API OpenAI cho {task_name}: {e}")
            return None
