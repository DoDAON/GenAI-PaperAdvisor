# 논문 PDF 임베딩 기반 RAG 평가 서비스

## 1. 서비스 개요
- **목표**: 논문 PDF를 임베딩 벡터화하여 ChromaDB에 저장하고, 사용자가 업로드한 논문과의 유사도 측정을 통해 상위 3개의 유사 논문을 기반으로 생성형 AI가 논문 평가 및 개선 제안을 제공하는 서비스
- **주요 기술**: 
  - Python, Streamlit (웹 인터페이스)
  - ChromaDB (벡터 데이터베이스)
  - OpenAI Embedding (텍스트 임베딩)
  - Gemini Pro (생성형 AI)
  - PyPDF (PDF 텍스트 추출)

## 2. 프로젝트 구조
```
project-root/
├── main.py                # Streamlit 메인 애플리케이션
├── batch_process_pdfs.py  # PDF 일괄 처리 스크립트
├── paper_db.py            # ChromaDB 연동 및 논문 관리
├── embedding.py           # OpenAI 임베딩 API 연동
├── pdf_utils.py           # PDF 텍스트 추출 유틸리티
├── ai_eval.py             # Gemini Pro 평가 및 제안
├── requirements.txt       # 프로젝트 의존성
├── .env                   # 환경 변수 (API 키 등)
├── .gitignore            # Git 제외 파일 목록
├── data/
│   └── papers/           # 논문 PDF 저장소
├── chromadb_data/        # ChromaDB 벡터 DB
└── README.md
```

## 3. 주요 기능

### 3.1 논문 PDF 일괄 처리 (batch_process_pdfs.py)
- `data/papers/` 디렉토리의 모든 PDF 파일 자동 처리
- 텍스트 추출, 임베딩 생성, 메타데이터 추출
- ChromaDB에 자동 저장
- 실행 방법:
  ```bash
  python batch_process_pdfs.py
  ```

### 3.2 웹 인터페이스 (main.py)
1. **데이터 확인**
   - ChromaDB에 저장된 논문 수 확인
   - 데이터 없을 경우 처리 방법 안내

2. **유사 논문 검색**
   - 사용자 논문 PDF 업로드
   - 유사도 기반 Top-3 논문 검색
   - 각 논문의 메타데이터 표시

3. **AI 평가 및 제안**
   - Gemini Pro 기반 논문 평가
   - 유사 논문과의 비교 분석
   - 구체적인 개선 제안 제공

## 4. 설치 및 실행 방법

### 4.1 환경 설정
1. Python 3.8 이상 설치
2. 가상환경 생성 및 활성화:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
3. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

### 4.2 API 키 설정
`.env` 파일 생성 후 다음 환경변수 설정:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 4.3 실행 방법
1. 논문 PDF 일괄 처리:
   ```bash
   python batch_process_pdfs.py
   ```
2. 웹 서비스 실행:
   ```bash
   streamlit run main.py
   ```

## 5. 데이터베이스 구조 (ChromaDB)
- **Collection: papers**
  - **id**: UUID 기반 고유 식별자
  - **embedding**: OpenAI 임베딩 벡터
  - **metadata**:
    - title: 논문 제목
    - authors: 저자
    - year: 연도
    - abstract: 초록
    - source: 원본 파일명

## 6. 주의사항
- PDF 파일은 `data/papers/` 디렉토리에 저장
- API 키는 반드시 `.env` 파일에 설정
- 대용량 PDF 처리 시 메모리 사용량 주의
- 개인정보 및 저작권 이슈 주의

## 7. 향후 개선 사항
- [ ] 논문 메타데이터 자동 추출 개선
- [ ] 평가 이력 관리 기능
- [ ] 논문 태그/키워드 자동 추출
- [ ] 성능 최적화 및 에러 처리 강화
- [ ] UI/UX 개선