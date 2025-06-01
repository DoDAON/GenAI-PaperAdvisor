import os
import re
import logging
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text):
    """정확한 토큰 수 계산"""
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    return len(encoding.encode(text))

def preprocess_text(text):
    """텍스트 전처리 함수"""
    logger.info("텍스트 전처리 시작")
    # 불필요한 공백 제거
    text = ' '.join(text.split())
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', ' ', text)
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    logger.info("텍스트 전처리 완료")
    return text.strip()

def split_text(text, max_tokens=4000):  # 더 작은 청크 크기로 설정
    """텍스트를 토큰 제한에 맞게 분할하는 함수"""
    # 먼저 문단으로 분할
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 문단이 너무 길면 문장 단위로 더 분할
        if count_tokens(para) > max_tokens:
            sentences = re.split(r'[.!?]+', para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # 문장이 너무 길면 단어 단위로 분할
                if count_tokens(sentence) > max_tokens:
                    words = sentence.split()
                    temp_chunk = []
                    temp_length = 0
                    
                    for word in words:
                        word_tokens = count_tokens(word)
                        if temp_length + word_tokens > max_tokens:
                            if temp_chunk:
                                chunks.append(' '.join(temp_chunk))
                            temp_chunk = [word]
                            temp_length = word_tokens
                        else:
                            temp_chunk.append(word)
                            temp_length += word_tokens
                    
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                else:
                    # 문장을 현재 청크에 추가
                    sentence_tokens = count_tokens(sentence)
                    if current_length + sentence_tokens > max_tokens:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_tokens
        else:
            # 문단을 현재 청크에 추가
            para_tokens = count_tokens(para)
            if current_length + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [para]
                current_length = para_tokens
            else:
                current_chunk.append(para)
                current_length += para_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # 각 청크의 토큰 수 확인 및 필요시 추가 분할
    final_chunks = []
    for chunk in chunks:
        if count_tokens(chunk) > max_tokens:
            # 청크가 여전히 너무 크면 더 작게 분할
            words = chunk.split()
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_tokens = count_tokens(word)
                if temp_length + word_tokens > max_tokens:
                    if temp_chunk:
                        final_chunks.append(' '.join(temp_chunk))
                    temp_chunk = [word]
                    temp_length = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_length += word_tokens
            
            if temp_chunk:
                final_chunks.append(' '.join(temp_chunk))
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def get_embedding(text, model="text-embedding-3-small"):
    """텍스트의 임베딩을 생성하는 함수"""
    logger.info("임베딩 생성 시작")
    # 텍스트 전처리
    text = preprocess_text(text)
    
    # 토큰 수 확인
    token_count = count_tokens(text)
    logger.info(f"전체 토큰 수: {token_count}")
    
    # 텍스트가 너무 길면 분할
    if token_count > 4000:  # 더 작은 임계값으로 설정
        logger.info("텍스트가 너무 길어 분할 처리 시작")
        chunks = split_text(text)
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_tokens = count_tokens(chunk)
                logger.info(f"청크 {i+1}/{len(chunks)} 처리 중: {chunk_tokens} 토큰")
                
                response = client.embeddings.create(
                    model=model,
                    input=chunk
                )
                embeddings.append(response.data[0].embedding)
                logger.info(f"청크 {i+1} 임베딩 생성 완료")
            except Exception as e:
                logger.error(f"청크 {i+1} 처리 중 오류 발생: {str(e)}")
                logger.error(f"청크 길이: {len(chunk)} 문자, 토큰 수: {count_tokens(chunk)}")
                raise e
                
        # 모든 청크의 임베딩 평균 계산
        logger.info("모든 청크의 임베딩 평균 계산 중")
        result = [sum(x) / len(x) for x in zip(*embeddings)]
        logger.info("임베딩 생성 완료")
        return result
    else:
        logger.info("단일 청크 임베딩 생성 중")
        response = client.embeddings.create(
            model=model,
            input=text
        )
        logger.info("임베딩 생성 완료")
        return response.data[0].embedding 
