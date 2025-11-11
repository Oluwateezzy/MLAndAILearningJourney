from typing import Dict, List
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import os


class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def process_document(self, document_paths: List[str]) -> list[dict]:
        documents = []
        for path in document_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                text = self._process_pdf(path)
            elif ext == ".docx":
                text = self._process_docx(path)
            elif ext == ".txt":
                text = self._process_txt(path)
            else:
                continue
            embedding = self.embedding_model.encode(text).tolist()
            documents.append({"text": text, "embedding": embedding})
        return documents

    def _process_pdf(self, path: str) -> str:
        text = ""
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    def _process_docx(self, path: str) -> str:
        text = ""
        doc = docx.Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip()

    def _process_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()


class VectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="documents")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, chunks: List[str], metadata: List[Dict]):
        embeddings = self.embedder.encode(chunks).tolist()
        for i, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                embeddings=[embeddings[i]],
                metadatas=[metadata[i]],
                ids=[str(i)],
            )

    def retrieve_relevant_context(self, query: str, n_results: int = 5) -> List[str]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )
        return [doc for doc in results["documents"][0]]


if __name__ == "__main__":
    processor = DocumentProcessor()
    vector_store = VectorStore()

    # Step 1: Process a document
    # docs = processor.process_document(["sample.txt"])

    # Step 2: Add document text into the vector store
    # for i, doc in enumerate(docs):
    #     vector_store.add_documents(
    #         chunks=[doc["text"]], metadata=[{"source": f"doc_{i}"}]
    #     )

    docs = processor.process_document(
        ["nigeria_facts.txt", "canada_facts.txt", "germany_facts.txt"]
    )
    for i, doc in enumerate(docs):
        vector_store.add_documents([doc["text"]], [{"source": f"doc_{i}"}])

    # Step 3: Ask a question (query)
    # query = "What is the capital of Nigeria?"
    # query = "When did Nigeria gain independence?"
    query = "What city is the capital of Germany?"
    results = vector_store.retrieve_relevant_context(query)

    print("\n--- Retrieved Documents ---")
    for i, res in enumerate(results):
        print(f"{i+1}. {res}")
