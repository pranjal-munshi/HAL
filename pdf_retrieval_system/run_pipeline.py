import nltk
nltk.download('stopwords')

from processors.pdf_chunker import PDFSubtopicProcessor
from indexing.faiss_manager import build_index, FAISSRetriever
from retriever.query_module import RetrieveHelicopterManualInfo
from sentence_transformers import SentenceTransformer

processor = PDFSubtopicProcessor()
processor.process_pdf_directory("pdfs")
chunk_metadata = processor.get_chunk_metadata()

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [chunk["text"] for chunk in chunk_metadata]
index, _ = build_index(texts, model)

retriever = FAISSRetriever(index, model, chunk_metadata, k=3)
query_module = RetrieveHelicopterManualInfo(retriever)

question = "Engine failure on landing"
result = query_module(question)

print(result)
