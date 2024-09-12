import json

json_list = [
   #  "Fire-keywords-by-date.json",
    "Fires-keywords-count-in-top10-content.json",
    "Fires-keywords-count-in-top10-title.json",
    "Fires-top-100-keywords-in-content.json",
    "Fires-top-100-keywords-in-title.json"
]

for file in json_list:
    print(file)
    with open("./"+file, 'r', encoding='utf8') as A:
        jsoncode = A.read()
    print(jsoncode)
    decoded = json.loads(jsoncode)
    with open("./"+file, "w", encoding='utf8') as A:
        json.dump(decoded, A, indent=2, ensure_ascii=False)