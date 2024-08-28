import os
from transformers import RagSequenceForGeneration, RagTokenizer

# 환경 변수 설정
os.environ["TRANSFORMERS_CACHE"] = "./cache"  # 현재 디렉토리의 cache 폴더를 사용하도록 설정

# 모델 로드
model_name = "facebook/rag-sequence-nq"
# model = RagSequenceForGeneration.from_pretrained(model_name)
# tokenizer = RagTokenizer.from_pretrained(model_name)

# 모델 경로 수정
model = RagSequenceForGeneration.from_pretrained(model_name, cache_dir="./cache")
tokenizer = RagTokenizer.from_pretrained(model_name, cache_dir="./cache")