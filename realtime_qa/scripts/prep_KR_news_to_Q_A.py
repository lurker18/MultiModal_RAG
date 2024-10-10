import os
import json
import glob
import random
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = api_key)

base_path = "/home/utopiamath/work/MultiModal_RAG/realtime_qa/scripts/"


def load_prompts(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return json.load(f)
def generate_question_from_news(news_content, news_date=None, prompts=None):
    selected_prompt = random.choice(prompts)
    prompt = selected_prompt.format(news=news_content, news_date=news_date)


    korean_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    korean_answer = korean_completion.choices[0].message.content.strip()

    return korean_answer

def generate_question_from_news00(news_content, news_date=None, prompts=None):
    selected_prompt = random.choice(prompts)
    prompt = selected_prompt.format(news=news_content, news_date=news_date)


    english_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    english_answer = english_completion.choices[0].message.content.strip()
    print("english_answer", english_answer)


    korean_translation_prompt = f"Translate the following to Korean:\n{prompt}"

    korean_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": korean_translation_prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    korean_answer = korean_completion.choices[0].message.content.strip()

    return english_answer, korean_answer
def format_question_data(question_id, question_date, source, url, sentence_kor, choices_kor, answer_kor, evidence_kor):
    return {
        "question_id": question_id,
        "question_date": question_date,
        "question_source": source,
        "question_url": url,
        "question_sentence": sentence_kor,
        "choices": choices_kor,
        "answer": [str(answer_kor)],
        "evidence": evidence_kor
    }
def format_ques5645tion_data(question_id, question_date, source, url, sentence_eng, sentence_kor, choices_eng, choices_kor, answer_eng, answer_kor, evidence_eng, evidence_kor):
    return {
        "question_id": question_id,
        "question_date": question_date,
        "question_source": source,
        "question_url": url,
        "question_sentence_eng": sentence_eng,
        "question_sentence_kor": sentence_kor,
        "choices_eng": choices_eng,
        "choices_kor": choices_kor,
        "answer_eng": [str(answer_eng)],
        "answer_kor": [str(answer_kor)],
        "evidence_eng": evidence_eng,
        "evidence_kor": evidence_kor
    }

def save_to_json(question_data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(question_data, f, ensure_ascii=False, indent=4)

def main(keyword):

    file_list = glob.glob(f"{base_path}*{keyword}*/*.json")[:3]

    prompts = load_prompts("prompts.json")

    for file in file_list:
        file_name = f"generated_questions_{keyword}_{os.path.basename(file).replace('.json','')}.jsonl"

        with open(file, "r", encoding="utf-8") as f:
            news_datas = json.load(f)

            for idx,news_data in enumerate(news_datas):
                print("idx", idx)

                if idx >10:
                    break
                if "item" in news_data and "documentList" in news_data["item"]:
                    document_list = news_data["item"]["documentList"]

                    for idx, document in enumerate(document_list):
                        if "entertain" in document["url"]:
                            continue
                        news_content = document["content"]
                        news_date = document.get("date", None)
                        news_title = document["title"]
                        news_url = document.get("url", "")
                        news_source = document.get("writerName", "Unknown")
                        evidence = news_content[:200]


                        generated_question_kor = generate_question_from_news(news_content, news_date, prompts)

                        choices_kor = ["선택 A", "선택 B", "선택 C", "선택 D"]
                        answer_kor = random.randint(0, 3)

                        question_id = f"{news_date}_{idx}_nota"
                        question_data = format_question_data(
                            question_id=question_id,
                            question_date=datetime.strptime(news_date, "%Y%m%d").strftime("%Y/%m/%d"),
                            source=news_source,
                            url=news_url,
                            sentence_kor=generated_question_kor,
                            choices_kor=choices_kor,
                            answer_kor=answer_kor,
                            evidence_kor=evidence
                        )

                        save_to_jsonl(question_data, file_name)
def save_to_jsonl(question_data, file_name):
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(json.dumps(question_data, ensure_ascii=False) + "\n")

def main2132():
    file_list = glob.glob(f"{base_path}*/*.json")[:2]

    prompts = load_prompts("prompts.json")

    for file in file_list:
        all_questions = []
        file_name = f"generated_questions_{os.path.basename(file)}"

        with open(file, "r", encoding="utf-8") as f:
            news_datas = json.load(f)

            for news_data in news_datas:
                if "item" in news_data and "documentList" in news_data["item"]:
                    document_list = news_data["item"]["documentList"]

                    for idx, document in enumerate(document_list):
                        news_content = document["content"]
                        news_date = document.get("date", None)
                        news_title = document["title"]
                        news_url = document.get("url", "")
                        news_source = document.get("writerName", "Unknown")
                        evidence = news_content[:200]


                        generated_question_eng, generated_question_kor = generate_question_from_news(news_content, news_date, prompts)

                        choices_eng = ["Choice A", "Choice B", "Choice C", "Choice D"]
                        choices_kor = ["선택 A", "선택 B", "선택 C", "선택 D"]
                        answer_eng = random.randint(0, 3)
                        answer_kor = answer_eng

                        question_id = f"{news_date}_{idx}_nota"
                        question_data = format_question_data(
                            question_id=question_id,
                            question_date=datetime.strptime(news_date, "%Y%m%d").strftime("%Y/%m/%d"),
                            source=news_source,
                            url=news_url,
                            sentence_eng=generated_question_eng,
                            sentence_kor=generated_question_kor,
                            choices_eng=choices_eng,
                            choices_kor=choices_kor,
                            answer_eng=answer_eng,
                            answer_kor=answer_kor,
                            evidence_eng=evidence,
                            evidence_kor=evidence
                        )

                        all_questions.append(question_data)


                        if len(all_questions) % 1 == 0:
                            save_to_json(all_questions, file_name)


        save_to_json(all_questions, file_name)
import re
import json


def parse_choices_from_question(question_data):

    choice_pattern = r'Choices:\s*(\d+\.\s*[^\n]+(?:\n\d+\.\s*[^\n]+)*)'
    match = re.search(choice_pattern, question_data["question_sentence"])

    if match:

        choices_text = match.group(1)


        choices = re.findall(r'\d+\.\s*([^\n]+)', choices_text)


        question_data["question_sentence"] = re.sub(choice_pattern, '', question_data["question_sentence"]).strip()


        question_data["choices"] = choices


    answer_pattern = r'Answer:\s*(\d+)\.\s*([^\n]+)'
    answer_match = re.search(answer_pattern, question_data["question_sentence"])

    if answer_match:

        answer_index = answer_match.group(1)
        question_data["answer"] = [answer_index]


        question_data["question_sentence"] = re.sub(answer_pattern, '', question_data["question_sentence"]).strip()

    return question_data


def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:

            question_data = json.loads(line)


            updated_data = parse_choices_from_question(question_data)


            outfile.write(json.dumps(updated_data, ensure_ascii=False) + '\n')


import re

def process_jsonl_text_clean(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()


    content = content.replace('", "choices": ["선택 A", "선택 B", "선택 C", "선택 D', '')



    content = re.sub(r'1\. ', '?", "choices": ["', content)
    content = re.sub(r'(\n)(\d+)\. ', r'", "\2": "', content)
    content = re.sub(r'2\.\s', '', content)
    content = re.sub(r'3\.\s', '', content)


    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


if __name__ == "__main__":
    keyword = "01_disaster_Fire_3years"
    main(keyword)

    input_jsonl = f'{base_path}{keyword}/001.jsonl'
    output_jsonl = f'{base_path}{keyword}/001_fixmcqlist.jsonl'
    output2_jsonl = f'{base_path}{keyword}/001_fixmcqlist_2.jsonl'
    process_jsonl(input_jsonl, output_jsonl)

    process_jsonl_text_clean(output_jsonl, output2_jsonl)
