import os
import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from pydantic import BaseModel
print("--- 1. FastAPI 앱 초기화 시작 ---")
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 켜질 때 실행
    print("서버 시작 중: DB 연결 확인...")
    try:
        # DB 테이블 생성 코드가 있다면 이 안으로 이동
        # Base.metadata.create_all(bind=engine)
        print("DB 연결 성공!")
    except Exception as e:
        print(f"DB 연결 실패: {e}")
    
    yield
    
    # 서버가 꺼질 때 실행
    print("서버 종료")

app = FastAPI(lifespan=lifespan)
# 1. static 폴더 안의 파일들(css, js 등)을 웹에서 접근할 수 있게 열어줍니다.
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. 메인 주소("/")로 접속했을 때 static/index.html을 반환합니다.
@app.get("/")
def read_root():
    return FileResponse("static/index.html")
print("--- 2. DB 엔진 설정 시작 ---")
# --- 1. Supabase PostgreSQL DB 연결 ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fallback.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL (Supabase) 연결 설정에 sslmode 추가
    engine = create_engine(
        DATABASE_URL, 
        pool_pre_ping=True, 
        pool_recycle=300,
        connect_args={"sslmode": "require"}
    )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 2. DB 테이블 모델 정의 ---
class BannedWord(Base):
    __tablename__ = "banned_words"
    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, unique=True, nullable=False)

class SystemPrompt(Base):
    __tablename__ = "system_prompt"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)

class UserDevice(Base):
    __tablename__ = "user_devices"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True, nullable=False)
    violations = Column(Integer, default=0)
    is_blocked = Column(Boolean, default=False)


try:
    Base.metadata.create_all(bind=engine)
    def init_system_prompt():
        db = SessionLocal()
        try:
            if db.query(SystemPrompt).count() == 0:
                db.add(SystemPrompt(content="너는 친절하고 유용한 학교 AI 도우미 챗봇이야. 학생들이 물어보는 질문에 정중하게 답변해줘."))
                db.commit()
        finally:
            db.close()
    init_system_prompt()
    print("✅ 데이터베이스 연결 및 초기화 성공!")
except Exception as db_err:
    print(f"⚠️ 데이터베이스 연결 실패 (서버는 켜집니다): {db_err}")

# --- Helper 함수들 ---
def get_banned_words():
    db = SessionLocal()
    try:
        words = db.query(BannedWord.word).all()
        return [w[0] for w in words]
    finally:
        db.close()

def get_system_prompt():
    db = SessionLocal()
    try:
        prompt = db.query(SystemPrompt).order_by(SystemPrompt.id.desc()).first()
        return prompt.content if prompt else "너는 친절하고 유용한 학교 AI 도우미 챗봇이야."
    finally:
        db.close()

# [최적화] DB를 단 1번만 열어서 금지어 포함 여부 검사
def check_profanity(text: str) -> bool:
    banned_words = get_banned_words()
    for word in banned_words:
        if word in text:
            return True
    return False

def handle_device_violation(device_id: str, is_violation: bool):
    db = SessionLocal()
    try:
        device = db.query(UserDevice).filter(UserDevice.device_id == device_id).first()
        if not device:
            device = UserDevice(device_id=device_id, violations=0, is_blocked=False)
            db.add(device)
            db.commit()
            db.refresh(device)

        if device.is_blocked:
            return True, device.violations

        if is_violation:
            device.violations += 1
            if device.violations >= 10:
                device.is_blocked = True
            db.commit()

        return device.is_blocked, device.violations
    finally:
        db.close()

# --- 3. 기본 메인 페이지 ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html 파일을 찾을 수 없습니다.</h1>"

# --- 4. 챗봇 API (/chat) ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

class ChatRequest(BaseModel):
    message: str
    device_id: str = "unknown_device"

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    device_id = request.device_id.strip()
    user_message = request.message.strip()

    # 1. 기기 차단 여부 확인
    is_blocked, current_violations = handle_device_violation(device_id, is_violation=False)
    if is_blocked:
        return {"response": "이용 약관 위반으로 인해 귀하의 계정은 일시 제한되었습니다. 나중에 다시 시도해주세요."}

    # 2. 사용자 메시지 금지어 체크
    if check_profanity(user_message):
        is_now_blocked, new_violations = handle_device_violation(device_id, is_violation=True)
        if is_now_blocked:
            return {"response": "이용 약관 위반으로 인해 귀하의 계정은 일시 제한되었습니다. 나중에 다시 시도해주세요."}
        else:
            return {"response": "죄송합니다. 현재 제 범위를 벗어난 질문입니다. 다른 이야기를 해볼까요?"}

    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY가 설정되지 않았습니다.")

    # 3. DeepSeek API 호출
    current_system_prompt = get_system_prompt()
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": current_system_prompt},
            {"role": "user", "content": user_message}
        ]
    }

    try:
        response = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=15)
        if response.status_code != 200:
            return {"response": "AI 응답을 가져오는 데 실패했습니다."}
            
        bot_reply = response.json()["choices"][0]["message"]["content"]

        # 4. AI 답변 금지어 2차 체크
        if check_profanity(bot_reply):
            return {"response": "죄송합니다. 제 답변 중에 부적절한 내용이 포함되어 있었습니다. 다른 주제로 이야기해볼까요?"}

        return {"response": bot_reply}

    except Exception:
        return {"response": "서버 오류가 발생했습니다."}

# --- 5. 관리자 API ---
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "my-school-secret-1234")

class PromptUpdateRequest(BaseModel):
    prompt: str

@app.get("/admin/system-prompt")
def read_system_prompt():
    return {"system_prompt": get_system_prompt()}

@app.post("/admin/system-prompt")
def update_system_prompt(request: PromptUpdateRequest, x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    db = SessionLocal()
    try:
        new_prompt = SystemPrompt(content=request.prompt.strip())
        db.add(new_prompt)
        db.commit()
        return {"message": "시스템 프롬프트가 성공적으로 변경되었습니다.", "updated_prompt": request.prompt}
    finally:
        db.close()

@app.get("/admin/banned-words")
def list_banned_words():
    return {"banned_words": get_banned_words()}

@app.post("/admin/banned-words")
def add_banned_word(word: str, x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    db = SessionLocal()
    try:
        word_clean = word.strip()
        existing = db.query(BannedWord).filter(BannedWord.word == word_clean).first()
        if existing:
            return JSONResponse({"message": "이미 존재하는 금지어입니다."}, status_code=400)
        new_word = BannedWord(word=word_clean)
        db.add(new_word)
        db.commit()
        return {"message": f"'{word_clean}' 금지어가 성공적으로 추가되었습니다."}
    finally:
        db.close()

@app.delete("/admin/banned-words")
def delete_banned_word(word: str, x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    db = SessionLocal()
    try:
        word_clean = word.strip()
        target = db.query(BannedWord).filter(BannedWord.word == word_clean).first()
        if not target:
            return JSONResponse({"message": "존재하지 않는 금지어입니다."}, status_code=404)
        db.delete(target)
        db.commit()
        return {"message": f"'{word_clean}' 금지어가 삭제되었습니다."}
    finally:
        db.close()

@app.get("/admin/blocked-devices")
def get_blocked_devices(x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    db = SessionLocal()
    try:
        devices = db.query(UserDevice).all()
        return [{"device_id": d.device_id, "violations": d.violations, "is_blocked": d.is_blocked} for d in devices]
    finally:
        db.close()

@app.post("/admin/unblock-device")
def unblock_device(device_id: str, x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="권한이 없습니다.")
    db = SessionLocal()
    try:
        device = db.query(UserDevice).filter(UserDevice.device_id == device_id).first()
        if not device:
            return JSONResponse({"message": "해당 기기를 찾을 수 없습니다."}, status_code=404)
        device.violations = 0
        device.is_blocked = False
        db.commit()
        return {"message": f"기기('{device_id}') 차단이 해제되었습니다."}
    finally:
        db.close()