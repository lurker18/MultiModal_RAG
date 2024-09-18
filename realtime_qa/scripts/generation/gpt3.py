
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import string, datetime
import numpy as np
from utils.tools import add_today, cos_vector
import tiktoken

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = api_key)

def run_gpt3(questions, retrieved_data = None, generate = False, model = 'gpt-3.5-turbo-instruct', rm_date_q = False, rm_date_r = False):
    answers = []
    scores = []
    tokenizer = tiktoken.get_encoding("cl100k_base") # Default: cl100k_base == GPT-3.5
    #"gpt-4o"       : "o200k_base"   # e.g., gpt-4o
    #"gpt-4"        : "cl100k_base"  # e.g., gpt-4
    #"gpt-3.5-turbo": "cl100k_base"  # e.g., gpt-3.5-turbo
    
    model = 'gpt-3.5-turbo-instruct'
    for q_idx in range(len(questions)):
        question = questions[q_idx]
        if retrieved_data is not None:
            retrieved_text = get_retrieved_text(retrieved_data[q_idx], top_k = 5, rm_date_r = rm_date_r)
        else:
            retrieved_text = None
        if generate:
            answer, score = gpt3_question_gen(question, retrieved_text, model = model, rm_date_q = rm_date_q, tokenizer = tokenizer)
        else:
            if question.get("choices") and "list" in str(type(question.get("choices"))):
                answer, score = gpt3_question_choice(question, retrieved_text, model = model, rm_date_q = rm_date_q, tokenizer = tokenizer)
            else:
                answer, score = gpt3_question(question, retrieved_text, model = model, rm_date_q = rm_date_q, tokenizer = tokenizer)
        answers.append(answer)
        scores.append(score)
    return answers, scores

def gpt3_question(question, retrieved_text = None, model = 'gpt-3.5-turbo-instruct', rm_date_q = False, tokenizer = None):

    sentence = question["question_sentence"]
    if not rm_date_q:
        sentence = add_today(sentence, question["question_date"])
    prompt = "Question: " + sentence
    choices = question["choices"]
    prompt += "\n"
    for alphabet, choice in zip(string.ascii_uppercase, choices):
        prompt += "{}) {}\n".format(alphabet, choice)
    if retrieved_text is not None:
        # insert retrieved text
        prompt = retrieved_text + "\n" + prompt

    scores = []
    for alphabet, choice in zip(string.ascii_uppercase, choices):
        if "None of" in choice:
            ans = "No answer is in ({})".format(",".join(choices[:-1]))
        else:
            ans = "Assumed answer: {}) {}".format(alphabet, choice)

        ans_len = len(tokenizer.encode(ans)) - 1
        query = prompt + "\n" + ans
        if "None of" not in choice:
            query = query + "\n" + "Is assumed answer is right?"
        output = client.completions.create(
                                model = model,
                                prompt = query,
                                max_tokens = 1,
                                logprobs = 5,
                                # echo = True,
                                temperature = 0.05,
                                )

        lprobs = np.array(output.choices[0].logprobs.token_logprobs[-ans_len:])
        score = lprobs.mean()
        scores.append(score)
    scores = np.array(scores)
    answer = scores.argmax()
    probs = np.exp(scores)
    probs = probs/probs.sum()
    prob = probs[answer]
    return [str(answer)], str(prob)

def gpt3_question_choice(question, retrieved_text = None, model = 'gpt-3.5-turbo-instruct', rm_date_q = False, tokenizer = None):

    sentence = question["question_sentence"]
    choices = question["choices"]
    if not rm_date_q:
        sentence = add_today(sentence, question["question_date"])
    query =  f"""The assistant receives a question and four choices with some evidence. Please answer the number (0, 1, 2 or 3) of given choices that matches the question: 
    for example, given the choices ["A", "C", "Z", "None of the above"] and the question is given as "What is the last alphabet?", the assistant should answer '2', indicating the index of the answer 'Z'.
    Given question : {question["question_sentence"]}
    Given choices : {question["choices"]}
    Hint for Anser : {question["evidence"]}
    Answer :"""

    output = client.completions.create(

                            model = model,
                            prompt = query,
                            logprobs = 5,
                            #echo = True,
                            temperature = 0.01,
                            )
    lprobs = np.array(output.choices[0].logprobs.token_logprobs)
    score = lprobs.mean()
    answer = output.choices[0].text.strip()
    # 답변에서 생성하기
    if "list" in str(type(question["choices"])):
        if answer[0] in [str(j) for j in range(0, 10)]:
            answer = answer[0]
        elif answer[-1] in [str(j) for j in range(0, 10)]:
            answer = answer[-1]
        else:
            answer = "3"

    prob = np.exp(score)
    return [str(answer)], str(prob)

def gpt3_question_gen(question, retrieved_text = None, model = 'gpt-3.5-turbo-instruct', rm_date_q = False, tokenizer = None):

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
    # query = prompt + "\nAnswer:"
    # multiple choice answer
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

    output = client.completions.create(

                            model = model,
                            prompt = query,
                            logprobs = 5,
                            #echo = True,
                            temperature = 0.05,
                            )

    answer = output.choices[0].text.strip()
    if "list" in str(type(question["choices"])):
        if answer[0] in [str(j) for j in range(0, 10)]:
            answer = answer[0]
        elif answer[-1] in [str(j) for j in range(0, 10)]:
            answer = answer[-1]
        else:
            answer = "3"

    scores = np.array(output.choices[0].logprobs.token_logprobs)
    score = np.exp(scores.mean())
    with open("./query_write.txt", "w", encoding="utf8") as X:
        X.write('EOF\n')
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

        first_paraph = " ".join(content.split("\n\n")[:2])
        if "title" in article.keys():
            first_paraph = article["title"] + " " + first_paraph
        if not rm_date_r:
            retrieved_text += "Article on {}: {}\n".format(date, first_paraph)
        else:
            retrieved_text += "Article: {}\n".format(first_paraph)
    return retrieved_text
