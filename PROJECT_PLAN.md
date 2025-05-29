# 논문 PDF 임베딩 기반 RAG 평가 서비스 설계 및 개발 명세

## 1. 서비스 개요
- **목표**: 다량의 논문 PDF를 임베딩 벡터화하여 ChromaDB에 저장하고, 사용자가 직접 업로드한 논문 PDF와의 유사도 측정을 통해 상위 3개의 유사 논문을 기반으로 생성형 AI가 논문 평가 및 개선 제안을 제공하는 서비스 구축
- **주요 기술**: Python, Streamlit(또는 FastAPI), ChromaDB, Voyage Embedding, Gemini Pro(생성형 AI), PyPDF, LangChain 등

---

## 2. 전체 아키텍처

```
[논문 PDF 업로드/수집] → [텍스트 추출] → [임베딩 벡터화] → [ChromaDB 저장]
      ↑                                                        ↓
[사용자 논문 PDF 업로드] → [텍스트 추출] → [임베딩] → [유사도 검색(Top-3)]
                                                        ↓
                [유사 논문 + 사용자 논문 → 프롬프트 구성]
                                                        ↓
                        [Gemini Pro로 평가/개선 제안]
                                                        ↓
                                [결과 제공]
```

---

## 3. 데이터베이스(ChromaDB) 구조
- **Collection: papers**
  - **id**: 논문 고유 식별자
  - **embedding**: 논문 본문 임베딩 벡터
  - **metadata**:
    - title: 논문 제목
    - authors: 저자
    - year: 연도
    - abstract: 초록
    - source: 원본 파일명/경로 등

---

## 4. 주요 기능 및 개발 단계

### 1) 논문 PDF 등록 및 임베딩
- PDF 파일 업로드/수집
- PyPDF 등으로 텍스트 추출
- Voyage 임베딩 API로 벡터화
- ChromaDB에 저장 (메타데이터 포함)

### 2) 사용자 논문 업로드 및 유사도 검색
- 사용자 PDF 업로드
- 텍스트 추출 및 임베딩
- ChromaDB에서 코사인 유사도 기반 Top-3 논문 검색

### 3) 생성형 AI 평가/제안
- Top-3 논문과 사용자 논문을 프롬프트로 구성
- Gemini Pro API로 논문 평가 및 개선 제안 생성
- 결과 요약 및 시각화

### 4) 부가 기능(선택)
- 논문 메타데이터 자동 추출(제목, 저자 등)
- 평가/첨삭 이력 관리
- 논문별 태그/키워드 추출

---

## 5. 폴더 및 파일 구조 예시
```
project-root/
├── main.py                # Streamlit/FastAPI 메인 엔트리
├── requirements.txt       # 의존성 목록
├── paper_db.py            # ChromaDB 연동 및 논문 관리
├── embedding.py           # 임베딩 함수 (Voyage API)
├── pdf_utils.py           # PDF 텍스트 추출 유틸
├── ai_eval.py             # Gemini 프롬프트 및 평가 함수
├── data/
│   └── papers/            # 논문 PDF 저장소
├── chromadb_data/         # ChromaDB 벡터 DB
└── README.md
```

---

## 6. 개발 단계별 체크리스트
1. 프로젝트 환경 세팅 및 의존성 설치
2. PDF 텍스트 추출 기능 구현 및 테스트
3. Voyage 임베딩 연동 및 ChromaDB 저장
4. 사용자 논문 업로드 및 유사도 검색 구현
5. Gemini 프롬프트 설계 및 평가/제안 기능 구현
6. UI/UX(웹 또는 API) 개발
7. 테스트 및 배포

---

## 7. API/함수 설계 예시
- `extract_text_from_pdf(pdf_path) -> str`
- `get_embedding(text) -> List[float]`
- `add_paper_to_db(text, metadata)`
- `search_similar_papers(embedding, top_n=3) -> List[dict]`
- `evaluate_paper(user_paper, similar_papers) -> str`

---

## 8. 프롬프트 설계 예시 (Gemini)
```
[역할] 너는 논문 평가 및 개선 제안 전문가야.
[입력] 아래는 사용자가 제출한 논문과 유사한 3개의 논문이야.
- 각 논문의 장단점, 차이점, 개선 방향을 비교 분석해줘.
- 논문 구조, 논리성, 독창성, 연구 방법, 결론의 설득력 등도 평가해줘.
- 마지막엔 구체적인 개선 제안을 bullet point로 정리해줘.

[유사 논문 1]
...
[유사 논문 2]
...
[유사 논문 3]
...
[사용자 논문]
...
```

---

## 9. 기타 참고 사항
- 논문 PDF의 품질(스캔본/텍스트본)에 따라 텍스트 추출 정확도가 달라질 수 있음
- Voyage 임베딩은 대용량 논문에도 적합
- Gemini 프롬프트는 토큰 한도 내에서 요약/발췌 필요
- 개인정보 및 저작권 이슈 주의 