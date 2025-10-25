import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.openai import OpenAI
from typing import List
import hashlib
from dotenv import load_dotenv

load_dotenv()


class SimpleEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        # Simple hash-based embedding - 384 dimensions
        hash_obj = hashlib.sha256(text.encode())
        hex_str = hash_obj.hexdigest()

        # Generate 384 dimensions from the hash
        result = []
        for i in range(384):
            # Use modulo to cycle through the hex string
            idx = (i * 2) % len(hex_str)
            hex_pair = hex_str[idx:idx + 2]
            if len(hex_pair) == 2:
                result.append(float(int(hex_pair, 16)) / 255.0)
            else:
                result.append(0.5)  # Default value

        return result

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(text) for text in texts]


# Set up embedding model
Settings.embed_model = SimpleEmbedding()

# Set up LLM - trick it by saying it's gpt-3.5-turbo but using DeepSeek endpoint
Settings.llm = OpenAI(
    model="gpt-4o-mini",  # Use recognized name for validation
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    temperature=0,
    max_tokens=4096,
    additional_kwargs={"model": "deepseek-chat"}  # Actual model for DeepSeek
)


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index


nz_pdf = SimpleDirectoryReader("data").load_data()
nz_index = get_index(nz_pdf, "nz")
nz_engine = nz_index.as_query_engine()