import os
import uuid
from pathlib import Path
from pdf_utils import extract_text_from_pdf
from embedding import get_embedding
from paper_db import add_paper_to_db
import PyPDF2

def extract_metadata_from_pdf(pdf_path):
    """PDF 파일에서 메타데이터를 추출합니다."""
    metadata = {
        "id": str(uuid.uuid4()),
        "title": "",
        "authors": "",
        "year": "",
        "abstract": "",
        "source": os.path.basename(pdf_path)
    }
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.metadata:
                if pdf_reader.metadata.title:
                    metadata["title"] = pdf_reader.metadata.title
                if pdf_reader.metadata.author:
                    metadata["authors"] = pdf_reader.metadata.author
                if pdf_reader.metadata.creation_date:
                    # PDF 생성 날짜에서 연도 추출 시도
                    try:
                        year = pdf_reader.metadata.creation_date[2:6]
                        metadata["year"] = year
                    except:
                        pass
    except Exception as e:
        print(f"메타데이터 추출 중 오류 발생: {e}")
    
    return metadata

def process_pdf(pdf_path):
    """단일 PDF 파일을 처리합니다."""
    print(f"\n처리 중인 파일: {pdf_path}")
    
    # 텍스트 추출
    text = extract_text_from_pdf(pdf_path)
    print("✓ 텍스트 추출 완료")
    
    # 임베딩 생성
    embedding = get_embedding(text)
    print("✓ 임베딩 생성 완료")
    
    # 메타데이터 추출
    metadata = extract_metadata_from_pdf(pdf_path)
    print("✓ 메타데이터 추출 완료")
    
    # DB에 저장
    add_paper_to_db(embedding, metadata)
    print("✓ DB 저장 완료")
    
    return True

def main():
    # papers 디렉토리 경로
    papers_dir = Path("data/papers")
    
    # PDF 파일 목록 가져오기
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("처리할 PDF 파일이 없습니다. data/papers 디렉토리에 PDF 파일을 넣어주세요.")
        return
    
    print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다...")
    
    # 각 PDF 파일 처리
    for pdf_path in pdf_files:
        try:
            process_pdf(str(pdf_path))
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {pdf_path}")
            print(f"오류 내용: {e}")
            continue
    
    print("\n모든 파일 처리가 완료되었습니다!")

if __name__ == "__main__":
    main() 