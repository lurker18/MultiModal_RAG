## 파일 설명 

_dpr로 각 날짜별로 크롤링된 데이터에서 빈출어휘를 분석하고 질문 셋을 만들기 위해 제작했습니다.

* `(주제)-keywords-by-date.json` - 각 날짜별로 주제와 연관 있는 기사들의 총 개수, 제목에 자주 등장하는 키워드 top10, 본문에 자주 등장하는 키워드 top10 목록을 측정합니다.
* `(주제)-top-100-keywords-in-title.json` - 3년간 해당 주제에 기사 제목에 자주 등장하는 키워드 100개를 선정합니다. 
* `(주제)-top-100-keywords-in-content.json` - 3년간 해당 주제에 기사 본문에 자주 등장하는 키워드 100개를 선정합니다. 
* `(주제)-keywords-count-in-top10-title.json` - keywords-by-date.json 데이터 분석을 통해 3년동안 각 키워드마다 제목에서 자주 사용된 top10 리스트에 몇 번 들어갔는지 측정합니다.
* `(주제)-keywords-count-in-top10-content.json` - keywords-by-date.json 데이터 분석을 통해 3년동안 각 키워드마다 기사 본문에서 자주 사용된 top10 리스트에 몇 번 들어갔는지 측정합니다.