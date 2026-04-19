import pdfplumber
import fitz
import pytesseract
from PIL import Image
import numpy as np
import re
from collections import defaultdict


class PDFParser:
    def __init__(self, ocr_lang="eng+tur"):
        self.ocr_lang = ocr_lang

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------
    def parse(self, pdf_path):

        blocks = self.extract_with_pdfplumber(pdf_path)

        if self._is_text_poor(blocks):
            blocks = self.extract_with_pymupdf(pdf_path)

        if self._is_text_poor(blocks):
            blocks = self.extract_with_ocr(pdf_path)

        lines = self.group_into_lines(blocks)

        ordered_lines = self.reconstruct_reading_order(lines)

        text = "\n".join(ordered_lines)

        return self.normalize_text(text)

    # -----------------------------
    # PDFPLUMBER (IMPROVED)
    # -----------------------------
    def extract_with_pdfplumber(self, pdf_path):

        blocks = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                words = page.extract_words(
                    use_text_flow=True,
                    keep_blank_chars=False,
                    x_tolerance=2,
                    y_tolerance=3,
                )

                for w in words:
                    blocks.append(
                        {
                            "text": w["text"],
                            "x0": w["x0"],
                            "x1": w["x1"],
                            "top": w["top"],
                            "bottom": w["bottom"],
                            "page": page_num,
                        }
                    )

        return blocks

    # -----------------------------
    # PYMuPDF (IMPROVED)
    # -----------------------------
    def extract_with_pymupdf(self, pdf_path):

        blocks = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            data = page.get_text("dict")

            for block in data["blocks"]:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    text = " ".join(span["text"] for span in line["spans"])

                    if text.strip():
                        bbox = line["bbox"]

                        blocks.append(
                            {
                                "text": text,
                                "x0": bbox[0],
                                "x1": bbox[2],
                                "top": bbox[1],
                                "bottom": bbox[3],
                                "page": page_num,
                            }
                        )

        return blocks

    # -----------------------------
    # OCR (SAFE)
    # -----------------------------
    def extract_with_ocr(self, pdf_path):

        blocks = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            data = pytesseract.image_to_data(
                img, lang=self.ocr_lang, output_type=pytesseract.Output.DICT
            )

            for i in range(len(data["text"])):
                text = data["text"][i].strip()

                if text:
                    blocks.append(
                        {
                            "text": text,
                            "x0": data["left"][i],
                            "x1": data["left"][i] + data["width"][i],
                            "top": data["top"][i],
                            "bottom": data["top"][i] + data["height"][i],
                            "page": page_num,
                        }
                    )

        return blocks

    # -----------------------------
    # TEXT QUALITY CHECK
    # -----------------------------
    def _is_text_poor(self, blocks):

        if not blocks:
            return True

        total = sum(len(b["text"]) for b in blocks)

        return total < 400

    # -----------------------------
    # LINE GROUPING (FIXED)
    # -----------------------------
    def group_into_lines(self, blocks):

        if not blocks:
            return []

        # adaptive clustering threshold
        y_tolerance = self._estimate_y_tolerance(blocks)

        lines = defaultdict(list)

        for b in blocks:
            key = (b["page"], round(b["top"] / y_tolerance))

            lines[key].append(b)

        output = []

        for key in sorted(lines.keys()):
            line_blocks = sorted(lines[key], key=lambda x: x["x0"])

            text = self._join_words(line_blocks)

            output.append({"text": text, "page": key[0], "top": key[1]})

        return output

    # -----------------------------
    # ADAPTIVE TOLERANCE
    # -----------------------------
    def _estimate_y_tolerance(self, blocks):

        heights = [b["bottom"] - b["top"] for b in blocks]

        if not heights:
            return 5

        return max(4, np.median(heights))

    # -----------------------------
    # SMART WORD JOINING (FIXED)
    # -----------------------------
    def _join_words(self, blocks):

        if not blocks:
            return ""

        text = blocks[0]["text"]

        for i in range(1, len(blocks)):
            prev = blocks[i - 1]
            curr = blocks[i]

            gap = curr["x0"] - prev["x1"]

            # adaptive spacing (IMPORTANT FIX)
            if gap > 3:
                text += " " + curr["text"]
            else:
                text += curr["text"]

        return text

    # -----------------------------
    # READING ORDER (IMPROVED)
    # -----------------------------
    def reconstruct_reading_order(self, lines):

        pages = defaultdict(list)

        for l in lines:
            pages[l["page"]].append(l)

        output = []

        for page in sorted(pages.keys()):
            page_lines = pages[page]

            # better ordering: top + x fallback
            page_lines.sort(key=lambda x: (x["top"], x["text"]))

            output.extend([l["text"] for l in page_lines])

        return output

    # -----------------------------
    # NORMALIZATION (SAFE)
    # -----------------------------
    def normalize_text(self, text):

        text = text.lower()

        text = text.replace("ı", "i")

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
