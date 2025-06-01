import chromadb
from chromadb.utils import embedding_functions
import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_db.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info("ChromaDB 컬렉션 접근 시도")
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        logger.info(f"기존 컬렉션 '{COLLECTION_NAME}' 사용")
        return client.get_collection(COLLECTION_NAME, embedding_function=embedding_function)
    else:
        logger.info(f"새로운 컬렉션 '{COLLECTION_NAME}' 생성")
        return client.create_collection(COLLECTION_NAME, embedding_function=embedding_function)

def add_paper_to_db(embedding, metadata):
    logger.info(f"논문 추가 시작: {metadata.get('title', 'Unknown')}")
    collection = get_collection()
    doc_id = metadata.get("id")
    collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[doc_id]
    )
    logger.info(f"논문 추가 완료: {metadata.get('title', 'Unknown')}")

def search_similar_papers(embedding, top_n=3):
    logger.info(f"유사 논문 검색 시작 (top_n={top_n})")
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_n,
        include=["metadatas", "distances"]
    )
    logger.info("유사 논문 검색 완료")
    return results

def get_paper_count():
    """ChromaDB에 저장된 논문의 수를 반환합니다."""
    logger.info("저장된 논문 수 확인 시작")
    try:
        collection = get_collection()
        count = collection.count()
        logger.info(f"저장된 논문 수: {count}")
        return count
    except Exception as e:
        logger.error(f"논문 수 확인 중 오류 발생: {e}")
        return 0 