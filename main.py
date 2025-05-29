import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from pdf_utils import extract_text_from_pdf
from embedding import get_embedding
from paper_db import add_paper_to_db, search_similar_papers, get_paper_count
from ai_eval import generate_paper_feedback

load_dotenv()

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
            feedback = generate_paper_feedback(user_text, similar_papers)
            st.subheader("AI 평가 및 개선 제안 결과")
            st.markdown(feedback, unsafe_allow_html=True) 