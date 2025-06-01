import streamlit as st
import os
import uuid
import logging
from dotenv import load_dotenv
from pdf_utils import extract_text_from_pdf
from embedding import get_embedding
from paper_db import search_similar_papers, get_paper_count
from ai_eval import generate_paper_feedback
import anthropic

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Anthropic Claude API 설정
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def summarize_with_claude(text, system_prompt):
    logger.info("Claude API를 사용하여 텍스트 요약 시작")
    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"논문 내용:\n{text}"
                }
            ]
        )
        logger.info("텍스트 요약 완료")
        return message.content[0].text
    except Exception as e:
        logger.error(f"Claude API 호출 중 오류 발생: {e}")
        raise e

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
    logger.info("새로운 PDF 파일 업로드 감지")
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(search_file.read())
    logger.info(f"임시 파일 생성 완료: {temp_path}")
    
    logger.info("PDF에서 텍스트 추출 시작")
    user_text = extract_text_from_pdf(temp_path)
    logger.info("텍스트 추출 완료")
    
    logger.info("텍스트 임베딩 생성 시작")
    user_embedding = get_embedding(user_text)
    logger.info("임베딩 생성 완료")
    
    logger.info("유사 논문 검색 시작")
    results = search_similar_papers(user_embedding, top_n=3)
    logger.info("유사 논문 검색 완료")
    
    similar_papers = []
    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
        logger.info(f"유사 논문 {i+1} 처리 중: {meta['title']}")
        st.markdown(f"**{i+1}. {meta['title']}** (유사도: {1-dist:.4f})")
        st.markdown(f"- 저자: {meta['authors']}")
        st.markdown(f"- 연도: {meta['year']}")
        st.markdown(f"- 초록: {meta['abstract']}")
        st.markdown(f"- 파일명: {meta['source']}")
        st.markdown("---")
        similar_papers.append(meta)
    
    os.remove(temp_path)
    logger.info(f"임시 파일 삭제 완료: {temp_path}")

    st.header("2. 생성형 AI 논문 평가 및 개선 제안")
    if st.button("AI 평가 및 개선 제안 받기"):
        logger.info("AI 평가 프로세스 시작")
        with st.spinner("생성형 AI 평가 중..."):
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
                logger.info(f"논문 처리 중: {paper['source']}")
                paper_path = os.path.join("data/papers", paper['source'])
                if os.path.exists(paper_path):
                    paper_text = extract_text_from_pdf(paper_path)
                    similar_papers_text.append(paper_text)
                    logger.info(f"논문 {paper['source']} 요약 시작")
                    summary = summarize_with_claude(paper_text, summary_system_prompt)
                    summarized_papers.append(summary)
                    logger.info(f"논문 {paper['source']} 요약 완료")
            
            # 각 논문에 대한 요약 표시
            st.subheader("유사 논문 요약")
            for i, summary in enumerate(summarized_papers):
                with st.expander(f"논문 {i+1} 요약"):
                    st.markdown(summary)
            
            # AI 평가 및 개선 제안
            feedback = generate_paper_feedback(user_text, summarized_papers)
            st.subheader("AI 평가 및 개선 제안 결과")
            st.markdown(feedback, unsafe_allow_html=True) 