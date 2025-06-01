import os
import logging
from dotenv import load_dotenv
import anthropic

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_eval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

PROMPT_TEMPLATE = """
            # 지시문
            당신은 논문 평가 전문가 '김논평'입니다. 아래 제시된 사용자의 논문을 평가하고 개선 제안을 하는 것이 당신의 역할입니다.
            기존 공개 학위 논문 데이터를 참고하여 사용자의 논문을 개선해주세요.

            # 제약조건
            - **모든 출력은 Markdown을 적극 활용**하여 작성합니다.
            - 코드 블록은 사용하지 않습니다.
            - 단순히 문장을 다듬는 것이 아니라, 기존 학위 논문의 핵심 내용과 전략을 파악하여 이를 바탕으로 사용자 논문을 실질적으로 개선합니다.
            - 순서 서식을 사용하지 않고 최대한 문장의 형식으로 출력해야 합니다. 서술을 해야 합니다.

            ---

            # 사용자의 논문 텍스트 데이터
            다음은 평가 대상이 되는 사용자의 논문입니다:

            {{user_info_text}}

            ---

            # 참고할 학위 논문의 내용
            다음은 참고용 학위 논문입니다. 이를 참고하여 사용자의 논문을 개선해 주세요:

            {{summarized_document}}

            ---

            # 출력 형태

            ### 사용자 논문 분석 및 개선 방향 제안
            [사용자의 논문을 분석하고 , 제공된 학위 논문과 비교하여 전체적인 평가와 개선이 필요한 부분을 제시합니다. 구체적인 개선 방향을 제안합니다.]
            """

def generate_paper_feedback(user_text, summarized_papers):
    logger.info("논문 평가 및 피드백 생성 시작")
    
    # 프롬프트에 데이터 삽입
    logger.info("프롬프트 템플릿에 데이터 삽입")
    prompt = PROMPT_TEMPLATE.replace(
        "{{user_info_text}}", user_text
    ).replace(
        "{{summarized_document}}", "\n\n".join(summarized_papers)
    )
    
    logger.info("Claude API 호출 시작")
    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.3,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": "논문을 평가하고 개선 방향을 제안해주세요."
                }
            ]
        )
        logger.info("논문 평가 및 피드백 생성 완료")
        return message.content[0].text
    except Exception as e:
        logger.error(f"Claude API 호출 중 오류 발생: {e}")
        raise e 