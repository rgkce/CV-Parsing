import os
import json
from pathlib import Path
from tqdm import tqdm
from cv_parser_script.cv_parser8 import process_cv

def process_dataset(data_dir: str, output_file: str):
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))
    
    results = []
    
    for pdf_file in tqdm(pdf_files, desc="Parsing CVs"):
        try:
            cv_data = process_cv(pdf_file)
            # Remove internal confidence key
            cv_data.pop("__confidence__", None)
            results.append(cv_data)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\nSuccessfully processed {len(results)} CVs.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    process_dataset("data/cengdataset", "final_dataset.json")
