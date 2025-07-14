import os
import fitz
from datetime import datetime
from processors.extractor import SubtopicExtractor

class PDFSubtopicProcessor:
    def __init__(self):
        self.extractor = SubtopicExtractor()
        self.results_storage = {}

    def process_single_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        page_results = []

        for page_num in range(len(doc)):
            text = doc.load_page(page_num).get_text()
            if not text.strip():
                continue

            i = 0
            while i < len(text):
                start = max(0, i - 50)
                end = min(len(text), i + 512 + 50)
                chunk_text = text[start:end]
                subtopics = self.extractor.extract_all_methods(chunk_text)
                top_subtopics = subtopics[:10]
                page_results.append({
                    'page_number': page_num + 1,
                    'chunk_id': i,
                    'chunk_start': start,
                    'text': chunk_text,
                    'subtopics': subtopics,
                    'top_subtopics': top_subtopics
                })
                i += 512

        doc.close()
        pdf_name = os.path.basename(pdf_path)
        self.results_storage[pdf_name] = {
            'pdf_name': pdf_name,
            'pdf_path': pdf_path,
            'processed_at': datetime.now().isoformat(),
            'pages': page_results
        }

    def process_pdf_directory(self, directory="."):
        pdfs = [f for f in os.listdir(directory) if f.endswith(".pdf")]
        for pdf in pdfs:
            self.process_single_pdf(os.path.join(directory, pdf))

    def get_chunk_metadata(self):
        all_data = []
        for pdf_name, result in self.results_storage.items():
            for chunk in result['pages']:
                all_data.append({
                    'doc_name': pdf_name,
                    'page_number': chunk['page_number'],
                    'chunk_id': chunk['chunk_id'],
                    'chunk_start': chunk['chunk_start'],
                    'text': chunk['text'],
                    'subtopics': [s[0] for s in chunk['top_subtopics']]
                })
        return all_data
