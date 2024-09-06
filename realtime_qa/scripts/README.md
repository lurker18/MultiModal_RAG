# RealTime QA Scripts

All API keys are removed. You need to get your key for GPT-3 and Google custom search, if you want to use them.

## Retrieval
We run Google custom search (GCS) around the time the questions are available every week. To do so, run:
```bash
python retrieve_main.py --config gcs --in-file ./dummy_data/20220613_qa.jsonl --key <gcs_key> --engine <engine_key>
```
This will yield `dummy_data/2020201_gcs.jsonl`. Replace the `--in-file` argument with your questions.

## Answer Prediction (Reading Comprehension)
```bash
python baseline_main.py --in-file ../past/2022/20220617_qa.jsonl --config open_rag_gcs --gcs-file ../past/2022/20220617_gcs.jsonl
```
The six `config` choices are: `[closed_gpt3, closed_t5, open_gpt3_gcs, open_gpt3_dpr, open_rag_gcs, open_rag_dpr]`. See [our paper](https://arxiv.org/abs/2207.13332) for more details. Use `--generate` for generation.
* in-file : qa dataset for answer sheet
* config : models 
* gcs_file : retrived dataset 
* result : saved in '../baseline_results/'

## Evaluation
```bash
python evaluate_main.py --pred-file ../baseline_results/2022/20220617_qa_open_gpt3_gcs.jsonl --gold-file ../past/2022/20220617_qa.jsonl 
```
* pred-file : answer prediction result
* gold-file : original qa set for comarison
Result form
```
{'accuracy': (float)}
```

## File Data Format
* `baseline_results` - Answer Prediction Data (name format : qa + configuration)
* `latest`, `past` - Retrieved Data (name format : ..._gcs - gcs retriving, ..._dpr -> dpr retrieving)

## Parameter Settings
* `baseline_main.py` , `evaluate_main.py`, `retrieve_main.py`
```python 
os.environ["TRANSFORMERS_CACHE"] = '/home/utopiamath/.cache' # transformer cache path
os.environ['HF_HOME'] = '/home/utopiamath/.cache/huggingface' # huggingface download path
os.environ['HF_DATASET_CACHE'] = '/home/utopiamath/Downloads' # cache path
```
* set transformer cache, huggingface home and huggingface dataset cache (should be accessible)

## Warning ##
If you want to use `open_gpt3_gcs` option, you shoud adjust 