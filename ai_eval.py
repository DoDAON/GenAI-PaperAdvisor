import os
import logging
import time
from dotenv import load_dotenv
import anthropic

# log 디렉토리 생성
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'ai_eval.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# 개별 논문 평가를 위한 프롬프트
INDIVIDUAL_PAPER_PROMPT = """
            # 지시문
            당신은 논문 평가 전문가 '김논평'입니다. 아래 제시된 사용자의 논문과 참고 논문을 비교하여 개선 제안을 하는 것이 당신의 역할입니다.

            # 제약조건
            - **모든 출력은 Markdown을 적극 활용**하여 작성합니다.
            - 코드 블록은 사용하지 않습니다.
            - 참고 논문의 핵심 내용과 전략을 파악하여 사용자 논문의 개선점을 제시합니다.
            - 순서 서식을 사용하지 않고 최대한 문장의 형식으로 출력해야 합니다.

            ---

            # 사용자의 논문
            {{user_info_text}}

            ---

            # 참고 논문
            {{reference_paper}}

            ---

            # 출력 형태
            ### 참고 논문 기반 개선 제안
            [참고 논문을 바탕으로 사용자 논문의 개선점을 구체적으로 제시합니다.]
            """

# 최종 종합 평가를 위한 프롬프트
FINAL_EVAL_PROMPT = """
            # 지시문
            당신은 논문 평가 전문가 '김논평'입니다. 아래 제시된 개별 평가 결과들을 종합하여 최종 평가와 개선 제안을 하는 것이 당신의 역할입니다.

            # 제약조건
            - **모든 출력은 Markdown을 적극 활용**하여 작성합니다.
            - 코드 블록은 사용하지 않습니다.
            - 개별 평가 결과들을 종합하여 일관된 최종 평가를 제시합니다.
            - 순서 서식을 사용하지 않고 최대한 문장의 형식으로 출력해야 합니다.

            ---

            # 개별 평가 결과
            {{individual_evaluations}}

            ---

            # 출력 형태
            ### 최종 논문 분석 및 개선 방향 제안
            [개별 평가 결과를 종합하여 전체적인 평가와 구체적인 개선 방향을 제시합니다.]
            """

def evaluate_single_paper(user_text, reference_paper, attempt=0):
    """단일 논문 평가 함수"""
    max_retries = 3
    retry_delay = 60

    try:
        prompt = INDIVIDUAL_PAPER_PROMPT.replace(
            "{{user_info_text}}", user_text
        ).replace(
            "{{reference_paper}}", reference_paper
        )

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.3,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": "논문을 평가하고 개선 방향을 제안해주세요."
                }
            ]
        )
        return message.content[0].text

    except Exception as e:
        if "rate_limit_error" in str(e) and attempt < max_retries - 1:
            wait_time = retry_delay * (attempt + 1)
            logger.warning(f"Rate limit 도달. {wait_time}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
            return evaluate_single_paper(user_text, reference_paper, attempt + 1)
        else:
            logger.error(f"단일 논문 평가 중 오류 발생: {e}")
            raise e

def generate_final_evaluation(individual_evaluations):
    """개별 평가 결과를 종합하여 최종 평가 생성"""
    try:
        prompt = FINAL_EVAL_PROMPT.replace(
            "{{individual_evaluations}}", "\n\n".join(individual_evaluations)
        )

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            temperature=0.3,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": "개별 평가 결과를 종합하여 최종 평가를 제시해주세요."
                }
            ]
        )
        return message.content[0].text

    except Exception as e:
        logger.error(f"최종 평가 생성 중 오류 발생: {e}")
        raise e

def generate_paper_feedback(user_text, summarized_papers):
    """논문 평가 및 피드백 생성 메인 함수"""
    logger.info("논문 평가 및 피드백 생성 시작")
    
    try:
        # 각 논문별 개별 평가 수행
        individual_evaluations = []
        for i, paper in enumerate(summarized_papers):
            logger.info(f"논문 {i+1} 평가 시작")
            evaluation = evaluate_single_paper(user_text, paper)
            individual_evaluations.append(evaluation)
            logger.info(f"논문 {i+1} 평가 완료")
            time.sleep(2)  # API 호출 간 간격 추가
        
        # 개별 평가 결과를 종합하여 최종 평가 생성
        logger.info("최종 평가 생성 시작")
        final_evaluation = generate_final_evaluation(individual_evaluations)
        logger.info("최종 평가 생성 완료")
        
        return final_evaluation

    except Exception as e:
        logger.error(f"논문 평가 프로세스 중 오류 발생: {e}")
        raise e 