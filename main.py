import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from pdf_utils import extract_text_from_pdf
from embedding import get_embedding
from paper_db import search_similar_papers, get_paper_count
from ai_eval import generate_paper_feedback
import google.generativeai as genai

load_dotenv()

# Gemini API 설정
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def summarize_with_gemini(text, system_prompt):
    generation_config = {
        "temperature": 0.3,  # 더 결정적인(deterministic) 출력을 위해 낮은 temperature 사용
        "max_output_tokens": 1000,  # 출력 토큰 수 제한
    }
    
    prompt = f"{system_prompt}\n\n논문 내용:\n{text}"
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    return response.text

st.set_page_config(page_title="논문 RAG 평가 프로토타입", page_icon=":books:")
st.title("논문 PDF 임베딩 및 유사 논문 검색 (프로토타입)")

# ChromaDB에 저장된 논문 수 확인
paper_count = get_paper_count()

if paper_count == 0:
    st.warning("""
    ⚠️ 임베딩된 논문 데이터가 없습니다!
    
    다음 단계를 따라주세요:
    1. `data/papers/` 디렉토리에 PDF 파일들을 넣어주세요
    2. 터미널에서 다음 명령어를 실행해주세요:
       ```
       python batch_process_pdfs.py
       ```
    3. 처리가 완료되면 이 페이지를 새로고침해주세요
    """)
    st.stop()
else:
    st.success(f"✅ {paper_count}개의 논문이 임베딩되어 있습니다.")

st.header("1. 유사 논문 검색 및 AI 평가")
search_file = st.file_uploader("유사도 검색 및 평가용 논문 PDF 업로드", type=["pdf"], key="search")
if search_file is not None:
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(search_file.read())
    user_text = extract_text_from_pdf(temp_path)
    user_embedding = get_embedding(user_text)
    results = search_similar_papers(user_embedding, top_n=3)
    st.subheader("유사 논문 Top-3 결과:")
    similar_papers = []
    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
        st.markdown(f"**{i+1}. {meta['title']}** (유사도: {1-dist:.4f})")
        st.markdown(f"- 저자: {meta['authors']}")
        st.markdown(f"- 연도: {meta['year']}")
        st.markdown(f"- 초록: {meta['abstract']}")
        st.markdown(f"- 파일명: {meta['source']}")
        st.markdown("---")
        similar_papers.append(meta)
    os.remove(temp_path)

    st.header("2. 생성형 AI 논문 평가 및 개선 제안")
    if st.button("AI 평가 및 개선 제안 받기"):
        with st.spinner("생성형 AI 평가 중..."):
            # 유사 논문들의 텍스트 추출 및 요약
            similar_papers_text = []
            summarized_papers = []
            
            # 요약을 위한 시스템 프롬프트
            summary_system_prompt = """
            # 지시문
            제시되는 학위 논문의 내용과 구조를 **짧게요약**하여 핵심만 추출하는것이 너의 역할이다. 

            # 제약조건
            - 논문의 구조를 파악하여, 전체 카테고리, 각 카테고리의 주된 서술을 추출한다..
            - 마크다운을 적극활용하며 **질문과 답변의 핵심 내용만 짧게 추출한다**
            - 내용 추출 시, 부가 설명은 최소화하고, 학위 논문의 핵심 요소를 드러낼 수 있도록 한다.
            - 부연설명은하지않는다.
            """
            
            for paper in similar_papers:
                paper_path = os.path.join("data/papers", paper['source'])
                if os.path.exists(paper_path):
                    paper_text = extract_text_from_pdf(paper_path)
                    similar_papers_text.append(paper_text)
                    # 각 논문 요약
                    summary = summarize_with_gemini(paper_text, summary_system_prompt)
                    summarized_papers.append(summary)
            
            # 각 논문에 대한 요약 표시
            st.subheader("유사 논문 요약")
            for i, summary in enumerate(summarized_papers):
                with st.expander(f"논문 {i+1} 요약"):
                    st.markdown(summary)
            
            # AI 평가 및 개선 제안
            feedback = generate_paper_feedback(user_text, summarized_papers)
            st.subheader("AI 평가 및 개선 제안 결과")
            st.markdown(feedback, unsafe_allow_html=True) 