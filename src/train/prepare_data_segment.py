import sys
import json
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.prompts import prompt_segment

def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process JSON data.")
    parser.add_argument("--input_path", type=str, help="Path to the input TXT files.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file.")
    return parser.parse_args()


def main():
    args = parser_arguments()
    input_path = Path(args.input_path)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data_llama = []
    for file in input_path.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        messages = ""
        ground_truth = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith(":"):
                ground_truth += "Speaker:" + "\n"
                continue
            else:
                ground_truth += line.strip() + "\n"
            
            messages += line.strip() + " "
        
        data_llama.append({
            "messages": [
                {
                    "role": "user",
                    "content": prompt_segment + "\n" + messages.strip()
                },
                {
                    "role": "assistant",
                    "content": ground_truth.strip()
                }
            ]
        })
    
    json.dump(data_llama, output_file.open("w", encoding="utf-8"), indent=4, ensure_ascii=False)
    print(f"Data has been written to {output_file}")

if __name__ == "__main__":
    main()
