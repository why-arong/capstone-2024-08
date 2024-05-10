import os
import json

# 입력 및 출력 경로 설정
INPUT_ROOT_DIR = "DUMMY3" 
OUTPUT_FILE = "output.txt"

# 출력 파일 열기
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    # 입력 디렉토리 탐색
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        for file in files:
            # .json 파일인 경우에만 처리
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                with open(json_path, "r", encoding="utf-8") as f_json:
                    data = json.load(f_json)
                    # script 키가 존재하는 경우에만 처리
                    if "script" in data:
                        script = data["script"]
                        if "text" in script and "id" in script:
                            script_text = script["text"]
                            script_id = script["id"]
                            # WAV 파일 경로 생성
                            wav_path = os.path.join(root, file.replace(".json", ".wav"))
                            # 텍스트와 ID를 "|"로 구분하여 출력 파일에 쓰기
                            f_out.write(f"{wav_path}|{script_id}|{script_text}\n")
