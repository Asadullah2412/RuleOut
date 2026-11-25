# pdf => text 
from docling.document_converter import DocumentConverter
from pathlib import Path
import time

def text_extractor(input_path:str,extracted_path:str):
    """
    extracts texts from pdf of a given folder 
    """
    total_start = time.perf_counter()

    input_dir = Path(input_path)
    output_dir = Path(extracted_path)

    convertor =DocumentConverter()

    print("files in the folder are : ")
    for file in input_dir.iterdir():
        file_start = time.perf_counter()

        print("Processing:", file.stem)
        # print("File Name:" , file.name)
        result = convertor.convert(file)
        save_path = output_dir / f"{file.stem}.md"

        file_elapsed = time.perf_counter() - file_start
        result.document.save_as_markdown(save_path)

        print(f"Finished {file.stem} in {file_elapsed:.2f} seconds\n")

    total_elapsed = time.perf_counter() - total_start
    print(
        f"All done! Total time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)"
    )


text_extractor(input_path="./pdf_2/",extracted_path="./extracted_md/")
