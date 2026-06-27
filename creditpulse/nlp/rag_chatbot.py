"""
Phase 4 — RAG chatbot for loan officers.
Retrieves relevant CBK regulatory text to answer policy questions.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DOCS_PATH = Path("data/raw/cbk_reports/")


class CreditPulseRAG:
    """
    Retrieval-Augmented Generation chatbot for loan officers.
    Uses CBK Annual Reports as the knowledge base.
    """

    def __init__(self, docs_path: Path = DOCS_PATH, model: str = "claude-3-haiku-20240307"):
        self.docs_path = docs_path
        self.model = model
        self._vectorstore = None
        self._llm = None

    def _build_vectorstore(self):
        try:
            from langchain_community.document_loaders import PyPDFDirectoryLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings

            loader = PyPDFDirectoryLoader(str(self.docs_path))
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self._vectorstore = Chroma.from_documents(chunks, embeddings)
            logger.info(f"Vectorstore built from {len(chunks)} chunks")
        except ImportError as e:
            logger.warning(f"RAG dependencies not installed: {e}")

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        if self._vectorstore is None:
            self._build_vectorstore()
        if self._vectorstore is None:
            return ["[CBK knowledge base not available — install langchain and chromadb]"]
        docs = self._vectorstore.similarity_search(query, k=k)
        return [d.page_content for d in docs]

    def answer(self, question: str) -> str:
        context_chunks = self.retrieve(question)
        context = "\n\n---\n\n".join(context_chunks)

        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                system=(
                    "You are a helpful assistant for loan officers in East Africa. "
                    "Answer questions about credit policy using the provided CBK regulatory context. "
                    "Be concise and cite specific regulations when relevant."
                ),
                messages=[{
                    "role": "user",
                    "content": f"Context from CBK reports:\n\n{context}\n\nQuestion: {question}"
                }],
            )
            return response.content[0].text
        except ImportError:
            return f"[LLM not available] Relevant context:\n\n{context}"
        except Exception as e:
            return f"Error: {e}\n\nRelevant context:\n\n{context}"
