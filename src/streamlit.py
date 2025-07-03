import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
from fpdf import FPDF


# Add the parent directory to the path to import from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocess.audio_preprocessing import CustomFeatureExtractor


from src.main import (
    preprocess_pdf,
    preprocess_audio,
    chat_gpt,
    load_model,
)

@st.cache_resource
def cached_load_model():
    """Loads and caches the models."""
    with st.spinner('Loading models...'):
        return load_model()

st.title("Heart Disease Information Processing System")

st.write(
    "Upload a PDF file with test results and an audio file of a conversation "
    "to get a summarized analysis."
)

pdf_file = st.file_uploader("Upload PDF file", type="pdf")
audio_file = st.file_uploader(
    "Upload audio file", type=["wav", "mp3", "m4a", "ogg"]
)

if pdf_file and audio_file:
    st.write("Files uploaded successfully. Processing...")

    # Create temporary files to store uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.getvalue())
        pdf_path = tmp_pdf.name

    file_extension = audio_file.type.split('/')[1]
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{file_extension}"
    ) as tmp_audio:
        tmp_audio.write(audio_file.getvalue())
        audio_path = tmp_audio.name

    try:
        # Load models
        pdf_process, audio_process = cached_load_model()

        # Process files
        st.info("Preprocessing PDF and audio files...")
        with st.spinner('Preprocessing files...'):
            conversation_preprocessed = preprocess_audio(
                audio_path, audio_process
            )
            test_result_preprocessed = preprocess_pdf(pdf_path, pdf_process)

        st.info(
            "Requesting ChatGPT for test result and conversation summaries..."
        )
        with st.spinner('Generating summary... This may take a moment.'):
            final_test_result = chat_gpt(
                test_result_preprocessed, type="test_result"
            )
            final_conversation = chat_gpt(
                conversation_preprocessed, type="conversation"
            )

        st.info("Combining final summaries...")
        final_summary = final_test_result + "\n" + final_conversation

        st.success("Processing complete!")
        st.subheader("Final Summary")
        st.text_area("Summary", final_summary, height=400)

        # Create PDF and add download button
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add UTF-8 font for broader character support
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.set_font('DejaVu', '', 12)

        pdf.multi_cell(0, 10, final_summary)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp_pdf_out:
            pdf.output(tmp_pdf_out.name)
            with open(tmp_pdf_out.name, "rb") as f:
                st.download_button(
                    label="Download Summary as PDF",
                    data=f.read(),
                    file_name="summary.pdf",
                    mime="application/pdf",
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Clean up temporary files
        os.remove(pdf_path)
        os.remove(audio_path)
