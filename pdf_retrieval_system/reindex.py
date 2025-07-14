from processors.pdf_chunker import PDFSubtopicProcessor
from indexing.faiss_manager import build_index, FAISSRetriever
from sentence_transformers import SentenceTransformer

def reindex_new_pdfs(pdf_directory, model=None):
    processor = PDFSubtopicProcessor()
    processor.process_pdf_directory(pdf_directory)
    metadata = processor.get_chunk_metadata()
    texts = [entry["text"] for entry in metadata]
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = build_index(texts, model)
    return index, model, metadata
