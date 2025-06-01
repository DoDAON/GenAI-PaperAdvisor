import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_text(text):
    """텍스트 전처리 함수"""
    # 불필요한 공백 제거
    text = ' '.join(text.split())
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', ' ', text)
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text(text, max_tokens=8000):
    """텍스트를 토큰 제한에 맞게 분할하는 함수"""
    # 간단한 분할 방법: 문단 단위로 분할
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        # 대략적인 토큰 수 계산 (영어 기준 1단어 = 1.3토큰)
        para_tokens = len(para.split()) * 1.3
        
        if current_length + para_tokens > max_tokens:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_tokens
        else:
            current_chunk.append(para)
            current_length += para_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def get_embedding(text, model="text-embedding-3-small"):
    """텍스트의 임베딩을 생성하는 함수"""
    # 텍스트 전처리
    text = preprocess_text(text)
    
    # 텍스트가 너무 길면 분할
    if len(text.split()) * 1.3 > 8000:  # 대략적인 토큰 수 체크
        chunks = split_text(text)
        embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(
                model=model,
                input=chunk
            )
            embeddings.append(response.data[0].embedding)
        # 모든 청크의 임베딩 평균 계산
        return [sum(x) / len(x) for x in zip(*embeddings)]
    else:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding 
