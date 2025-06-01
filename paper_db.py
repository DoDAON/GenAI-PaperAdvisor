import chromadb
from chromadb.utils import embedding_functions
import os

persist_directory = 'chromadb_data'
os.makedirs(persist_directory, exist_ok=True)

# OpenAI 임베딩 함수 설정
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path=persist_directory)

COLLECTION_NAME = "papers"

def get_collection():
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        return client.get_collection(COLLECTION_NAME, embedding_function=embedding_function)
    else:
        return client.create_collection(COLLECTION_NAME, embedding_function=embedding_function)

def add_paper_to_db(embedding, metadata):
    collection = get_collection()
    doc_id = metadata.get("id")
    collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[doc_id]
    )

def search_similar_papers(embedding, top_n=3):
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_n,
        include=["metadatas", "distances"]
    )
    return results

def get_paper_count():
    """ChromaDB에 저장된 논문의 수를 반환합니다."""
    try:
        collection = get_collection()
        return collection.count()
    except Exception as e:
        print(f"논문 수 확인 중 오류 발생: {e}")
        return 0 