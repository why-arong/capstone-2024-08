from fastapi import FastAPI, UploadFile
from feedback import levenshtein, text

app = FastAPI()


@app.post("/feedback/")
async def create_upload_file(file: UploadFile):
    # mac 에서 확인할 수 없어 일단 stt가 되었다고 가정
    guide = "지난 해 극장을 찾은 연간 관객 수가 역대 최대치를 기록했습니다"
    user = "지난 해 극장을 찾은 연관 관객 수가 역다 최다치를 기록했습니다"
    cleaned_guide = text._clean_text(guide, None)
    cleaned_user = text._clean_text(user, None)
    similarity_percentage = levenshtein.dist(cleaned_guide, cleaned_user)
    # similarity_percentage = levenshtein.dist(guide, user)
    return {"similarity_percentage": similarity_percentage}

