import json

"""
with open("Fires-top-100-keywords-in-title.json", "r", encoding="utf8") as A:
    jdt_list = json.load(A)

with open("Fires-top-100-keywords-in-content.json", "r", encoding="utf8") as A:
    jdc_list = json.load(A)


jdt_list = sorted(jdt_list, key=lambda x: x[1], reverse=True)
jdc_list = sorted(jdc_list, key=lambda x: x[1], reverse=True)

with open("Fires-top-100-keywords-in-title.json", "w", encoding="utf8") as A:
    jdt = {item[0]:item[1] for item in jdt_list}
    json.dump(jdt, A, ensure_ascii=False, indent=2)

with open("Fires-top-100-keywords-in-content.json", "w", encoding="utf8") as A:
    jdc = {item[0]: item[1] for item in jdc_list}
    json.dump(jdc, A, ensure_ascii=False, indent=2)
"""

"""
with open("Fire-keywords-by-date.json", "r", encoding="utf8") as A:
    js_r = json.load(A)

js_key = sorted(list(js_r.keys()))
js_r_new = {key:js_r[key] for key in js_key}

with open("Fire-keywords-by-date.json", "w", encoding="utf8") as A:
    json.dump(js_r_new, A, ensure_ascii=False, indent=2)
"""

with open("Fires-keywords-count-in-top10-title.json", "r", encoding="utf8") as A:
    jdt = json.load(A)

with open("Fires-keywords-count-in-top10-content.json", "r", encoding="utf8") as A:
    jdc = json.load(A)


jdt_list = [tuple(item) for item in jdt.items()]
jdc_list = [tuple(item) for item in jdc.items()]

jdt_list = sorted(jdt_list, key=lambda x: x[1], reverse=True)
jdc_list = sorted(jdc_list, key=lambda x: x[1], reverse=True)

with open("Fires-keywords-count-in-top10-title.json", "w", encoding="utf8") as A:
    jdt = {item[0]:item[1] for item in jdt_list}
    json.dump(jdt, A, ensure_ascii=False, indent=2)

with open("Fires-keywords-count-in-top10-content.json", "w", encoding="utf8") as A:
    jdc = {item[0]: item[1] for item in jdc_list}
    json.dump(jdc, A, ensure_ascii=False, indent=2)