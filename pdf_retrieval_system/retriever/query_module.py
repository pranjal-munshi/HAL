import dspy

class RetrieveHelicopterManualInfo(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever

    def forward(self, question):
        results = self.retriever.retrieve(question)
        top = results[0] if results else {}
        return {
            "doc_name": top.get("doc_name", ""),
            "page_number": top.get("page_number", -1),
            "chunk_id": top.get("chunk_id", -1),
            "page_content": top.get("text", "")[:1000],
            "subtopics": top.get("subtopics", [])
        }
