from markitdown import MarkItDown

structure_pdf = """
Patient information
Patient name: {name}
Date of birth: {date_of_birth} Age: {age} Gender: {gender}
Date of assessment: {date_of_assessment}
Physician name: {physician_name}

Clinical presentation
Chest pain (onset, duration, severity): {chest_pain}
Shortness of breath: {shortness_of_breath}
Nausea/vomiting: {nausea_vomiting}
Sweating (diaphoresis): {sweating}
Dizziness or syncope: {dizziness_syncope}
Risk factors: {risk_factors}

Tests
Findings: {findings}
ST-segment elevation: {st_segment_elevation} 
T-wave inversion: {t_wave_inversion}
Troponin I: {result_i}
Troponin T: {result_t}
"""

class PDFProcessor:
    def __init__(self, docintel_endpoint: str = "<document_intelligence_endpoint>"):
        self.md = MarkItDown(docintel_endpoint=docintel_endpoint)
    
    def convert(self, pdf_path: str) -> MarkItDown:
        result = self.md.convert(pdf_path)
        return result.text_content

    def restructure(self, text_parsed: str) -> str:
        lines = text_parsed.split("\n")
        if lines[38].strip() != "Tests":
            del lines[38]
        
        structured_pdf = structure_pdf.format(
            name=lines[9].strip(),
            date_of_birth=lines[5].split(":")[-1].strip(),
            age=lines[10].split(":")[-1].strip(),
            gender=lines[14].split(":")[-1].strip(),
            date_of_assessment=lines[11].strip(),
            physician_name=lines[12].strip(),
            chest_pain=lines[31].strip(),
            shortness_of_breath=lines[33].strip().split(" ")[-1],
            nausea_vomiting=lines[34].strip().split(" ")[-1],
            sweating=lines[35].strip().split(" ")[-1],
            dizziness_syncope=lines[36].strip().split(" ")[-1],
            risk_factors=lines[37].strip().split(" ")[-1],
            findings=lines[43].strip(),
            st_segment_elevation=lines[46].strip(),
            t_wave_inversion=lines[49].strip(),
            males_i=lines[54].split("Males: ")[-1].strip(),
            females_i=lines[58].split("Females: ")[-1].strip(),
            result_i=lines[63].strip(),
            males_t=lines[56].split("Males: ")[-1].strip(),
            females_t=lines[60].split("Females: ")[-1].strip(),
            result_t=lines[65].strip(),
        )
        
        return structured_pdf
