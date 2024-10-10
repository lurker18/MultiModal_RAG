
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import string, datetime
import numpy as np
from utils.tools import add_today, cos_vector
import tiktoken

from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
import re

load_dotenv()

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

MODEL_PATH = '/mnt/nvme01/huggingface/models/Vaiv' # Server
MODEL_NAMES = [
    'GeM2-Llamion-14B-Base',
    'GeM2-Llamion-14B-Chat',
    'GeM2-Llamion-14B-LongChat',
    'llamion-14b-base',
    'llamion-14b-chat'
]

selected_model = MODEL_PATH + '/' + MODEL_NAMES[0] # Gem2-Llamion-14b-chat 사용법
# selected_model = 'vaiv/GeM2-Llamion-14B-Base'

# 모델을 GPU로 이동
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def run_vaivgem(questions, retrieved_data = None, generate = False, model = selected_model, rm_date_q = False, rm_date_r = False):
    answers = []
    scores = []
    tokenizer = AutoTokenizer.from_pretrained(selected_model, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(selected_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    # model.generation_config = GenerationConfig.from_pretrained(selected_model)
    
    for q_idx in range(len(questions)):
        question = questions[q_idx]
        if retrieved_data is not None:
            retrieved_text = get_retrieved_text(retrieved_data[q_idx], top_k = 5, rm_date_r = rm_date_r)
        else:
            retrieved_text = None
        if generate:
            answer, score = vaivgem_question_gen(question, retrieved_text, model = model, rm_date_q = rm_date_q, tokenizer = tokenizer)
        else:
            answer, score = vaivgem_question(question, retrieved_text, model = model, rm_date_q = rm_date_q, tokenizer = tokenizer)
        answers.append(answer)
        scores.append(score)
    return answers, scores


def vaivgem_question(question, retrieved_text = None, model = selected_model, rm_date_q = False, tokenizer = None):

    sentence = question["question_sentence"]
    if not rm_date_q:
        sentence = add_today(sentence, question["question_date"])

    query =  f"""주어진 힌트를 참고한 뒤에 질문을 보고 선택지 중에  답변을 하세요. 선택지는 4개가 있으며, 첫 번째값이 정답이면 0, 두 번째 값이 정답이면1, 세 번째 값이 정답이면 2, 네 번째 값이 정답이면 3을 답하시면 됩니다. 힌트는 질문에 대해 답변을 할 때 참고할 수 있는 내용입니다.   
    질문 : {question["question_sentence"]}
    선택지 : {question["choices"]}
    힌트 : {question["evidence"]}
    답변 :"""

    token_inputs = tokenizer(query, return_tensors="pt").to(device)
    inputs_dic = {key:value for key, value in token_inputs.items() if key != 'token_type_ids'}

    with torch.no_grad():
        outputs = model.generate(**inputs_dic, max_length=500)
    
    output = outputs[0]

    # lprobs = np.array(output.choices[0].logprobs.token_logprobs)
    # score = lprobs.mean()
    answer = tokenizer.decode(output, skip_special_tokens=True)
    # 답변에서 생성하기
    if "list" in str(type(question["choices"])):
        # print(f"예시: {answer}")
        if re.search(r"답변\s?:\s?(\d)", answer):
            num = re.search(r"답변\s?:\s?(\d)", answer).group(1)
            answer = str(num)
            print(num)
        elif answer[0] in [str(j) for j in range(0, 10)]:
            answer = answer[0]
        elif answer[-1] in [str(j) for j in range(0, 10)]:
            answer = answer[-1]
        else:
            answer = "3"

    # prob = np.exp(score)
    prob = 0.5
    return [str(answer)], str(prob)


def vaivgem_question_gen(question, retrieved_text = None, model = selected_model, rm_date_q = False, tokenizer = None):

    sentence = question["question_sentence"]
    demo = "What is the capital city of Japan?"

    if not rm_date_q:
        demo = add_today(demo, question["question_date"])
        sentence = add_today(sentence, question["question_date"])

    prompt = ""
    prompt += "Question: " + sentence
    if retrieved_text is not None:
        # insert retrieved text
        prompt = retrieved_text + "\n" + prompt

    if "list" in str(type(question["choices"])):
        query =  f"""The assistant receives a question and four choices with some evidence. Please answer the number (0, 1, 2 or 3) of given choices that matches the question: 
    for example, given the choices ["A", "C", "Z", "None of the above"] and the question is given as "What is the last alphabet?", the assistant should answer '2', indicating the index of the answer 'Z'.
    Given question : {question["question_sentence"]}
    Given choices : {question["choices"]}
    Hint for Anser : {question["evidence"]}
    Answer :"""
    # simple answer
    else:
        query = prompt + "\nAnswer:"
    # query = prompt + "\nAnswer:"

    token_inputs = tokenizer(query, return_tensors="pt").to(device)
    inputs_dic = {key:value for key, value in token_inputs.items() if key != 'token_type_ids'}

    with torch.no_grad():
        outputs = model.generate(**inputs_dic, max_length=10)
    
    output = outputs[0]

    # answer = output["choices"][0]["text"].strip()
    answer = tokenizer.decode(output, skip_special_tokens=True)
    if "list" in str(type(question["choices"])):
        # print(f"예시: {answer}")
        if re.search(r"답변\s?:\s?(\d)", answer):
            num = re.search(r"답변\s?:\s?(\d)", answer).group(1)
            answer = str(num)
            print(num)
        if answer[0] in [str(j) for j in range(0, 10)]:
            answer = answer[0]
        elif answer[-1] in [str(j) for j in range(0, 10)]:
            answer = answer[-1]
        else:
            answer = "3"
    
    score = 0.5

    return answer, str(score)


def get_retrieved_text(retrieved_datum, top_k = 5, rm_date_r = False):
    search_result = retrieved_datum["search_result"]
    retrieved_text = ""
    for article in search_result[:top_k]:
        if "publish_date" not in article:
            continue
        date = article["publish_date"]
        content = article["text"]
        if content == '':
            continue
        date = datetime.datetime.strptime(date, '%Y/%m/%d')
        date = date.strftime("%B %d, %Y")
        #first_paraph = content.split("\n\n")[0]
        first_paraph = " ".join(content.split("\n\n")[:2])
        if "title" in article.keys():
            first_paraph = article["title"] + " " + first_paraph
        if not rm_date_r:
            retrieved_text += "Article on {}: {}\n".format(date, first_paraph)
        else:
            retrieved_text += "Article: {}\n".format(first_paraph)
    return retrieved_text
