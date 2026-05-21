from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from semantic_router.encoders import FastEmbedEncoder
import numpy as np


class LangChainFastEmbedBridge:
    """Bridges Layer 1 encoder to LangChain's vector store interface."""

    def __init__(self):
        self.encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encoder(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.encoder([text])[0]


class ToolVectorRegistry:
    def __init__(self):
        self.embeddings = LangChainFastEmbedBridge()
        self.db = None
        self._tools_source = []

    def register_tool_schema(self, name: str, description: str):
        """Adds a tool definition to the raw index source."""
        self._tools_source.append(Document(
            page_content=description,
            metadata={"tool_name": name}
        ))

    def build_index(self):
        """Compiles the tool definitions into an in-memory vector index."""
        if self._tools_source:
            self.db = FAISS.from_documents(self._tools_source, self.embeddings)
            print(f" Layer 2 Tool RAG compiled with {len(self._tools_source)} schemas.")

    def search_relevant_tools(self, user_query: str, k: int = 3) -> list[dict]:
        """Retrieves the top K tools closest to the user's intent."""
        if not self.db:
            return []
        results = self.db.similarity_search(user_query, k=k)
        return [{"name": doc.metadata["tool_name"], "description": doc.page_content} for doc in results]


# Global singleton for tool indexing
tool_rag_registry = ToolVectorRegistry()