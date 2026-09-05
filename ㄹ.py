import os
import json
import asyncio
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

BOTH_BLOCKED_WORDS = ["송연호", "촬영", "얼굴", '우정혁']
OUTPUT_ONLY_BLOCKED_WORDS = ["최호근", "홍윤건", "최준혁"]

SCHOOL_CONTEXT = """
너는 우리 학교 학생들을 돕는 AI 도우미야.
학교: 하안북중학교
점심시간: 오후 12:45-오후 1:35
주요 학사 일정:
2학기 3학년 1차 정기시험: 2026.11.17 - 18
체육 대회: 알려진 바 없음
하늘빛 축제: 2026.12.24
종업식 및 졸업식: 2027.1.3
- 매점: 매점 없음
체육관: 3층
미술실: 3층
음악실: 2층과 4층
급식실: 2층
가사실: 2층
도서실: 1층
세탁실: 1층
1학년 학년부장: 김혜진(도덕)
2학년 학년부장: 알 수 없음
3학년 학년부장: 변신옥(역사)
- 도서관: 최대 3권, 7일 대출 가능
"""
# 기기별 금지어 위반 횟수 저장소 (실제 운영 시 DB로 교체 가능)
# 구조: {"device_id": 위반_횟수}
VIOLATION_COUNTS: Dict[str, int] = {}
MAX_ALLOWED_VIOLATIONS = 8


class ChatRequest(BaseModel):
    message: str
    device_id: str  # 프론트엔드에서 고유 기기 ID 전달받음

def check_words(text: str, word_list: list) -> bool:
    for word in word_list:
        if word in text:
            return True
    return False

# 위반 횟수 증가 및 영구 차단 여부 반환 함수
def record_violation(device_id: str) -> int:
    current_count = VIOLATION_COUNTS.get(device_id, 0) + 1
    VIOLATION_COUNTS[device_id] = current_count
    return current_count

async def generate_chat_stream(user_message: str, device_id: str):
    all_output_blocked = BOTH_BLOCKED_WORDS + OUTPUT_ONLY_BLOCKED_WORDS

    # 0. 이미 10회 이상 위반한 기기인지 확인
    if VIOLATION_COUNTS.get(device_id, 0) >= MAX_ALLOWED_VIOLATIONS:
        yield json.dumps({
            "type": "banned", 
            "content": "이용 약관 위반으로 인해 귀하의 계정은 일시 정지되었습니다. 나중에 다시 시도해 주세요."
        }) + "\n"
        return

    # 1. 입력 검사
    if check_words(user_message, BOTH_BLOCKED_WORDS):
        count = record_violation(device_id)
        
        if count >= MAX_ALLOWED_VIOLATIONS:
            yield json.dumps({
                "type": "banned", 
                "content": "이용 약관 위반으로 인해 귀하의 계정은 일시 정지되었습니다. 나중에 다시 시도해 주세요."
            }) + "\n"
        else:
            yield json.dumps({
                "type": "blocked", 
                "content": f"죄송합니다. 현재 제 범위를 벗어난 질문입니다. 다른 주제로 이야기해볼까요?"
            }) + "\n"
        return

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SCHOOL_CONTEXT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            stream=True
        )

        accumulated_text = ""

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                new_token = chunk.choices[0].delta.content
                accumulated_text += new_token

                # 2. 실시간 출력 검사
                if check_words(accumulated_text, all_output_blocked):
                    count = record_violation(device_id)

                    if count >= MAX_ALLOWED_VIOLATIONS:
                        yield json.dumps({
                            "type": "banned", 
                            "content": "이용 약관 위반으로 인해 귀하의 계정은 일시 정지되었습니다. 나중에 다시 시도해 주세요."
                        }) + "\n"
                    else:
                        yield json.dumps({
                            "type": "blocked", 
                            "content":"죄송합니다. 제 답변 중에 부적절한 내용이 포함되어 있었습니다. 다른 주제로 이야기해볼까요?"
                        }) + "\n"
                    return

                yield json.dumps({"type": "token", "content": new_token}) + "\n"
                await asyncio.sleep(0.02)

    except Exception as e:
        yield json.dumps({"type": "error", "content": "API 호출 중 오류가 발생했습니다."}) + "\n"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message.strip()
    device_id = request.device_id.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")
    if not device_id:
        raise HTTPException(status_code=400, detail="잘못된 접근입니다.")

    return StreamingResponse(
        generate_chat_stream(user_message, device_id),
        media_type="application/x-ndjson"
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)