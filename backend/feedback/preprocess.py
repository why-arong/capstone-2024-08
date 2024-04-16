# import argparse
import text
# from utils import load_filepaths_and_text


if __name__ == '__main__':
    original_text = "지난해 극장을 찾은 연간 관객 수가 역대 최다치를 기록했습니다."
    cleaned_text = text._clean_text(original_text, None)
