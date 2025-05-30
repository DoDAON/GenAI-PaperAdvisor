import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 프롬프트 내용은 사용자가 직접 입력하도록 공란 처리
PROMPT_TEMPLATE = """"""

def generate_paper_feedback(user_text, similar_papers):
    # 유사 논문 정보 텍스트로 변환
    similar_texts = []
    for i, meta in enumerate(similar_papers, 1):
        similar_texts.append(f"[유사 논문 {i}]\n제목: {meta.get('title', '')}\n저자: {meta.get('authors', '')}\n연도: {meta.get('year', '')}\n초록: {meta.get('abstract', '')}")
    similar_block = "\n\n".join(similar_texts)

    # 실제 프롬프트는 사용자가 직접 입력
    prompt = PROMPT_TEMPLATE

    # Gemini API 호출
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([
        prompt,
        f"[사용자 논문]\n{user_text}",
        similar_block
    ])
    return response.text 