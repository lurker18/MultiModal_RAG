from konlpy.tag import Mecab
import os
import json

m = Mecab()

total_keywords_in_title = {}
total_keywords_num_in_contents = {}
# list_elem = ['NNG', 'NNP', 'VV', 'VA'] # 가장 많이 쓰는 품사 - 명사, 동사, 형용사
list_elem = ["NNG", "NNP"] # 일반명사, 고유명사만 남기기

json_dir = "./Fire/"
filelist = [file for file in os.listdir(json_dir) if file[-6:] == ".jsonl"] # jsonl 파일 모음
filelist = sorted(filelist)

result_by_date = {} # 날짜별 top10 키워드
top_10_keywords_title_count = {} # top10에 들어간 키워드 숫자 (title 기준)
top_10_keywords_count = {} # top10에 들어간 키워드 숫자 (content 기준)

file_count = len(filelist) # 전체 파일 수

for file in filelist:
    data_list = []
    keywords_in_title = {}
    keywords_num_in_contents = {}
    file_dir = json_dir + file
    with open(file_dir, 'r', encoding='utf8') as jsonl:
        for line in jsonl:
            data_list.append(json.loads(line))
    
    # pairs 이용해서 추가
    date_info = file.split("_")[0][-10:]
    result_by_date[date_info] = {"count": len(data_list), "title": {}, "content": {}}
    
    for data_dict in data_list:
        data_title = data_dict["title"]
        data_content = data_dict["content"]
        m_title = m.pos(data_title)
        m_content = m.pos(data_content)

        for tup in m_title:
            if tup[1] in list_elem:
                if keywords_in_title.get(tup[0]):
                    keywords_in_title[tup[0]] += 1
                else:
                    keywords_in_title[tup[0]] = 1
                if total_keywords_in_title.get(tup[0]):
                    total_keywords_in_title[tup[0]] += 1
                else:
                    total_keywords_in_title[tup[0]] = 1
        
        for tup in m_content:
            if tup[1] in list_elem:
                if keywords_num_in_contents.get(tup[0]):
                    keywords_num_in_contents[tup[0]] += 1
                else:
                    keywords_num_in_contents[tup[0]] = 1
                if total_keywords_num_in_contents.get(tup[0]):
                    total_keywords_num_in_contents[tup[0]] += 1
                else:
                    total_keywords_num_in_contents[tup[0]] = 1
        
        keywords_in_title_pairs = [tuple(item) for item in keywords_in_title.items()]
        keywords_num_in_contents_pairs = [tuple(item) for item in keywords_num_in_contents.items()]

        keywords_in_title_pairs = sorted(keywords_in_title_pairs, key=lambda x: x[1], reverse=True)
        keywords_num_in_contents_pairs = sorted(keywords_num_in_contents_pairs, key=lambda x: x[1], reverse=True)

        result_by_date[date_info]["title"] = {item[0]: item[1] for item in keywords_in_title_pairs[:10]}
        result_by_date[date_info]["content"] = {item[0]:item[1] for item in keywords_num_in_contents_pairs[:10]}


# 마지막으로 분석
total_keywords_in_title_pairs = [tuple(item) for item in total_keywords_in_title.items()]
total_keywords_num_in_contents_pairs = [tuple(item) for item in total_keywords_num_in_contents.items()]

# 3년간 전체 키워드 분석 top 100
total_keywords_in_title_pairs = sorted(total_keywords_in_title_pairs, key=lambda x: x[1], reverse=True)[:100]
total_keywords_num_in_contents_pairs = sorted(total_keywords_num_in_contents_pairs, key=lambda x: x[1], reverse=True)[:100]

total_keywords_in_title_pairs = {item[0]:item[1] for item in total_keywords_in_title_pairs}
total_keywords_num_in_contents_pairs = {item[0]: item[1] for item in total_keywords_num_in_contents_pairs}

# top10 빈도분석
for work_date, work_date_dict in result_by_date.items():
    date_title = work_date_dict["title"]
    date_content = work_date_dict["content"]
    for word in date_title:
        if top_10_keywords_title_count.get(word):
            top_10_keywords_title_count[word] += 1
        else:
            top_10_keywords_title_count[word] = 1
    for word in date_content:
        if top_10_keywords_count.get(word):
            top_10_keywords_count[word] += 1
        else:
            top_10_keywords_count[word] = 1

# top10 빈도분석 키워드 정렬용
top_10_keywords_title_count_pairs = [tuple(item) for item in top_10_keywords_title_count.items()]
top_10_keywords_count_pairs = [tuple(item) for item in top_10_keywords_count.items()]

top_10_keywords_title_count_pairs = sorted(top_10_keywords_title_count_pairs, key=lambda x: x[1], reverse=True)
top_10_keywords_count_pairs = sorted(top_10_keywords_count_pairs, key=lambda x: x[1], reverse=True)

top_10_keywords_title_count = {item[0]:item[1] for item in top_10_keywords_title_count_pairs}
top_10_keywords_count = {item[0]:item[1] for item in top_10_keywords_count_pairs}

# json 변환 시도

# 날짜별 키워드 분석
with open("Fire-keywords-by-date.json", "w", encoding='utf8') as A:
    json.dump(result_by_date, A, indent=2, ensure_ascii=False)

# 3년간 제목 top 100
with open("Fires-top-100-keywords-in-title.json", "w", encoding="utf8") as B:
    json.dump(total_keywords_in_title_pairs, B, indent=2, ensure_ascii=False)
    
# 3년간 내용 top 100
with open("Fires-top-100-keywords-in-content.json", "w", encoding="utf8") as B:
    json.dump(total_keywords_num_in_contents_pairs, B, indent=2, ensure_ascii=False)

# top 10 제목 횟수 분석
with open("Fires-keywords-count-in-top10-title.json", "w", encoding="utf8") as C:
    json.dump(top_10_keywords_title_count, C, indent=2, ensure_ascii=False)

# top 10 횟수 분석
with open("Fires-keywords-count-in-top10-content.json", "w", encoding="utf8") as C:
    json.dump(top_10_keywords_count, C, indent=2, ensure_ascii=False)
