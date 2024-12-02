import pandas as pd
import numpy as np
import json

from strong_reject.load_datasets import load_strongreject_small
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset, register_jailbreak
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset
from datasets import Dataset

def load_harmful_benchmark(HB_name):
    if HB_name == 'MI':
        # 读取/storage_fast/lhsheng/lhsheng/NeFL/dataset/MaliciousInstruct.txt
        with open('dataset/MaliciousInstruct.txt', 'r') as f:
            harmful_questions = f.readlines()
        # question_dict = {'MaliciousInstruct': []}
        question_list = []
        for q in harmful_questions:
            question_list.append({"forbidden_prompt": q})
        
        forbidden_prompt_dataset = Dataset.from_list(question_list)

    # if HB_name == 'XSTest':
    #     xstest = pd.read_csv('dataset/xstest.csv')
    #     question_list = []
    #     for idx, line in xstest.iterrows():
    #         question_list.append(line['type'], line['prompt'])

    if HB_name == 'SR-small':
        forbidden_prompt_dataset = load_strongreject_small()
    
    return forbidden_prompt_dataset


        
def load_templates():
    with open('dataset/jailbreak.json', 'r') as f:
        templates = json.load(f)
    
    template_list = []
    for k, v in templates.items():
        if '[INSERT PROMPT HERE]' in v['prompt']:
            template_list.append([k, v['prompt']])
    return template_list

def complete_prompt(prompt, template):
    prompt = template['prefix'] + prompt + template['suffix']
    return prompt