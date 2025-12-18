
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import tiktoken
import openai
import json
import os.path as osp
import asyncio
from asyncapi import process_api_requests_from_file
import torch
import logging
import ast
import matplotlib.pyplot as plt
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from openail.config import configs
from openail.utils import load_mapping
import re
import random
import numpy as np

async def call_async_api(request_filepath, save_filepath, request_url, api_key, max_request_per_minute, max_tokens_per_minute, sp, ss):
    await process_api_requests_from_file(
            requests_filepath=request_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_request_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name='cl100k_base',
            max_attempts=int(2),
            logging_level=int(logging.INFO),
            seconds_to_pause=sp,
            seconds_to_sleep=ss
        )

def openai_text_api(input_text, api_key, model_name = "gpt-3.5-turbo", temperature = 0, n = 1):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": input_text}],
        temperature=temperature,
        api_key=api_key,
        n = n)
    return response 

def generate_chat_input_file(input_text, model_name = 'gpt-3.5-turbo', temperature = 0, n = 1):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['model'] = model_name
        obj['messages'] = [
            {
                'role': 'user',
                'content': text 
            }
        ]
        obj['temperature'] = temperature
        obj['n'] = n
        jobs.append(obj)
    return jobs 


def generate_zero_shot_prompts(data, num_negative, args, data_categories_dir, example="", example_dict="", object_cat = "Paper", question = "W?", answer_format = ""):
    prompts = []
    label_names = [data.label_names[i] for i in args.known_class]
    # question = "Generate some negative samples?"
    for i in range(num_negative):
        # prompt = "{}: \n".format(object_cat)
        prompt = "Task: \n"
        prompt += "There are following listed Paper topics: \n"
        # with open(data_categories_dir, 'r') as f:
        #     positive_samples = f.readlines()   
        #     for positive_sample in positive_samples[1:-1]:
        #         in_context += "[" + positive_sample + "]" + "\n"     
        prompt += "[" + ", ".join(label_names) + "]" + "\n"
        prompt += "Generate a text with a Paper title and abstract outside the provided categories.  \
            " + "\n"
        # prompt += "\nOutput:\n"
        # prompt += examples + ", " + example_dict + "\n"  

        # prompt += "Question: {}. Generate a text with a topic outside the provided categories. For example,  \
        #     [{{\"answer\": <your_answer>}}]        \
        #     ".format(in_context)  

        # prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts                 

def generate_close_topics(data, topic, args, api_key):
    label_names = [data.label_fullnames[i] for i in args.known_class]
    topic_prompt = "Task: \n"
    topic_prompt += "There are following listed {} categories: \n".format(topic)
    topics_seen = "[" + ", ".join(label_names) + "]" + "\n"
    topic_prompt += topics_seen
    topic_seen_prompt = topic_prompt + "Which major {} category do these categories belong to? \n".format(topic)
    topic_seen_prompt += "Output : [{\"answer\": <your_answer>}] \n"

    response = openai_text_api(topic_seen_prompt, api_key, model_name = "gpt-4o", temperature = 0, n = 1)
    major_category_seen = [x['message']['content'] for x in response['choices']]
    try:
        dic_category_seen = json.loads(major_category_seen[0])        
    except:
        match = re.search(r'\[{"answer":.*?}\]', major_category_seen[0])
        if match:
            dic_category_seen = match.group()
            dic_category_seen = json.loads(dic_category_seen)
    major_category_seen = dic_category_seen[0].get("answer")
        
    topic_unseen_prompt = "Generate 20 possible {} categories that belong to {} but distinct from the provided categories {}.".format(topic, major_category_seen, topics_seen)
    topic_unseen_prompt += "Output : [{\"answer\": <your_answer>}, {\"answer\": <your_answer>}, ...] \n"
    
    response = openai_text_api(topic_unseen_prompt, api_key, model_name = "gpt-4o", temperature = 0.1, n = 1)
    topics_unseen = [x['message']['content'] for x in response['choices']]  
    try:
        dic_topics_unseen = json.loads(topics_unseen[0])
    except:
        topics_unseen_ = topics_unseen[0].strip("```json").strip("```").strip()
        dic_topics_unseen = json.loads(topics_unseen_)
    
    output_data = {
        "major_category_seen": major_category_seen,
        "dic_topics_unseen": dic_topics_unseen
    }    
    with open(args.dataset+"_"+str(args.known_class)+"_"+str(args.llm_model_method)+'_topics_unseen.json', 'w') as file:

        json.dump(output_data, file, indent=4)  
    
    return output_data

def generate_close_topics_nlp(label_names, args, api_key, object_cat="Paper"):

    topic_prompt = "Task: \n"
    topic_prompt += "There are following listed topics of {}: \n".format(object_cat)
    topics_seen = "[" + ", ".join(label_names) + "]" + "\n"
    topic_prompt += topics_seen
    topic_seen_prompt = topic_prompt + "Which major category do these themes belong to? \n"
    topic_seen_prompt += "Output : [{\"answer\": <your_answer>}] \n"

    response = openai_text_api(topic_seen_prompt, api_key, model_name = "gpt-3.5-turbo", temperature = 0, n = 1)
    major_category_seen = [x['message']['content'] for x in response['choices']]
    try:
        dic_category_seen = json.loads(major_category_seen[0])        
    except:          
        match = re.search(r'\[{"answer":.*?}\]', major_category_seen[0])
        if match:
            dic_category_seen = match.group()
            dic_category_seen = json.loads(dic_category_seen)
    major_category_seen = dic_category_seen[0].get("answer")
        
    topic_unseen_prompt = "Generate 20 possible topics that belong to {} but distinct from the provided topics {}.".format(major_category_seen, topics_seen)
    topic_unseen_prompt += "Output : [{\"answer\": <your_answer>}, {\"answer\": <your_answer>}, ...] \n"
    
    response = openai_text_api(topic_unseen_prompt, api_key, model_name = "gpt-4o", temperature = 0, n = 1)
    topics_unseen = [x['message']['content'] for x in response['choices']]  
    topics_unseen = topics_unseen[0].replace("```json", "").replace("```", "").strip()
    dic_topics_unseen = json.loads(topics_unseen)
    
    output_data = {
        "major_category_seen": major_category_seen,
        "dic_topics_unseen": dic_topics_unseen
    }    
    with open(args.dataset+"_"+str(args.known_class)+"_"+str(args.llm_model_method)+'_topics_unseen.json', 'w') as file:

        json.dump(output_data, file, indent=4)  
    
    return output_data

def generate_zero_shot_close_prompts(data, num_negative, topics_unseen, args, data_categories_dir, example="", example_dict="", object_cat = "Paper", question = "W?", answer_format = ""):
    prompts = []
    label_names = topics_unseen
    topic_prompt = "Task: \n"
    topic_prompt += "There are following listed Paper topics: \n"
    topic_prompt += "[" + ", ".join(label_names) + "]" + "\n"

    # question = "Generate some negative samples?"
    for i in range(num_negative):
        # prompt = "{}: \n".format(object_cat)
        prompt = "Task: \n"
        prompt += "There are following listed Paper topics: \n"
        # with open(data_categories_dir, 'r') as f:
        #     positive_samples = f.readlines()   
        #     for positive_sample in positive_samples[1:-1]:
        #         in_context += "[" + positive_sample + "]" + "\n"     
        prompt += "[" + ", ".join(label_names) + "]" + "\n"
        prompt += "Generate a text with a Paper title and abstract in the provided categories.  \
            " + "\n"
        # prompt += "\nOutput:\n"
        # prompt += examples + ", " + example_dict + "\n"  

        # prompt += "Question: {}. Generate a text with a topic outside the provided categories. For example,  \
        #     [{{\"answer\": <your_answer>}}]        \
        #     ".format(in_context)  

        # prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts                         


def generate_close_prompts_from_classes(data, num_negative, major_category_seen, topics_unseen, args, data_categories_dir, example="", example_dict="", object_cat = "Paper", question = "W?", answer_format = ""):
    prompts = []
    label_names = topics_unseen
    topic_prompt = "Task: \n"
    topic_prompt += "There are following listed Paper topics: \n"
    topic_prompt += "[" + ", ".join(label_names) + "]" + "\n"

    # question = "Generate some negative samples?"
    if args.prompt_method in ["zero_shot_from_closecategoty"]:
        prompt = "Task: \n"
        prompt += "There is following listed Paper topic: {} \n".format(major_category_seen)

        prompt += "Generate a text with a Paper title and abstract in the provided category.  \
            " + "\n"
        prompts.append(prompt) 
    else:       
        for i in label_names[:args.num_llm_dummy_class]:
            # prompt = "{}: \n".format(object_cat)
            prompt = "Task: \n"
            prompt += "There is following listed Paper topic: {} \n".format(i)
            # with open(data_categories_dir, 'r') as f:
            #     positive_samples = f.readlines()   
            #     for positive_sample in positive_samples[1:-1]:
            #         in_context += "[" + positive_sample + "]" + "\n"     
            # prompt += "[" + ", ".join(label_names) + "]" + "\n"
            prompt += "Generate a text with a Paper title and abstract in the provided category.  \
                " + "\n"
            # prompt += "\nOutput:\n"
            # prompt += examples + ", " + example_dict + "\n"  

            # prompt += "Question: {}. Generate a text with a topic outside the provided categories. For example,  \
            #     [{{\"answer\": <your_answer>}}]        \
            #     ".format(in_context)  

            # prompt += "\nOutput:\n"
            prompts.append(prompt)
    return prompts                         

def efficient_openai_text_api(input_text, input_file, output_file, api_key="change_this_to_your_key", model_name="gpt-3.5-turbo", request_url = "https://api.openai.com/v1/chat/completions", rewrite = True, temperature = 0, n = 1):
    # import ipdb; ipdb.set_trace()
    # openai_result = []
    rewrite = True
    non_empty_results = []
    results = []
    if not osp.exists(output_file):
        jobs = generate_chat_input_file(input_text, model_name=model_name, temperature = temperature, n = n)

        with open(input_file, "w") as f:
            for i, job in enumerate(jobs):
                json_string = json.dumps(job)
                if job['messages'][0]['content'] != "":
                    f.write(json_string + "\n")
                    non_empty_results.append(i)
        asyncio.run(
            call_async_api(
                input_file, save_filepath=output_file,
                request_url=request_url,
                api_key=api_key,
                max_request_per_minute=60, 
                max_tokens_per_minute=10000,
                sp=60,
                ss=0
            )
        )
    openai_result = []

    with open(output_file, 'r') as f:
        # import ipdb; ipdb.set_trace()
        for i, line in enumerate(f):
            json_obj = json.loads(line.strip())
            content = json_obj[1]
            idx = json_obj[-1]
            choices = []
            non_empty_results.append(i)
            if content == "":
                openai_result.append(("", idx))
                # import ipdb; ipdb.set_trace()
            elif isinstance(idx, int):
                choices = [x['message']['content'] for x in json_obj[1]['choices']]
                openai_result.append((choices, idx))
                # input = json_obj[0]['messages'][0]['content']
                # input = input.split('\n')[1]
                # texts_input.append(input)
            else:
                idx = json_obj[-2]
                new_result = openai_text_api(json_obj[0]['messages'][0]['content'], api_key, model_name = json_obj[0]['model'], temperature = json_obj[0]['temperature'], n = json_obj[0]['n'])
                choices = [x['message']['content'] for x in new_result['choices']]
                openai_result.append((choices, idx))
                # input = json_obj[0]['messages'][0]['content']
                # input = input.split('\n')[1]
                # texts_input.append(input)
    openai_result = sorted(openai_result, key=lambda x:x[-1])
    results = [("", idx) for idx in range(len(input_text))]
    # for i, r in enumerate(openai_result):
    #     results[non_empty_results[i]] = r
    return openai_result


def efficient_openai_text_ood(input_text, input_file, output_file, api_key="change_this_to_your_key", model_name="gpt-3.5-turbo", request_url = "https://api.openai.com/v1/chat/completions", rewrite = True, temperature = 0, n = 1):
    # import ipdb; ipdb.set_trace()
    # openai_result = []
    rewrite = False
    non_empty_results = []
    results = []
    if not osp.exists(output_file):
        jobs = generate_chat_input_file(input_text, model_name=model_name, temperature = temperature, n = n)

        with open(input_file, "w") as f:
            for i, job in enumerate(jobs):
                json_string = json.dumps(job)
                if job['messages'][0]['content'] != "":
                    f.write(json_string + "\n")
                    non_empty_results.append(i)
        asyncio.run(
            call_async_api(
                input_file, save_filepath=output_file,
                request_url=request_url,
                api_key=api_key,
                max_request_per_minute=60, 
                max_tokens_per_minute=10000,
                sp=60,
                ss=0
            )
        )
    openai_result = []
    with open(output_file, 'r') as f:
        # import ipdb; ipdb.set_trace()
        for i, line in enumerate(f):
            json_obj = json.loads(line.strip())
            content = json_obj[1]
            idx = json_obj[-1]
            choices = []
            non_empty_results.append(i)
            if content == "":
                openai_result.append(("", idx))
                # import ipdb; ipdb.set_trace()
            elif isinstance(idx, int):
                choices = [x['message']['content'] for x in json_obj[1]['choices']]
                openai_result.append((choices, idx))
            else:
                idx = json_obj[-2]
                new_result = openai_text_api(json_obj[0]['messages'][0]['content'], api_key, model_name = json_obj[0]['model'], temperature = json_obj[0]['temperature'], n = json_obj[0]['n'])
                choices = [x['message']['content'] for x in new_result['choices']]
                openai_result.append((choices, idx))
    openai_result = sorted(openai_result, key=lambda x:x[-1])
    results = [("", idx) for idx in range(len(input_text))]
    # for i, r in enumerate(openai_result):
    #     results[non_empty_results[i]] = r
    return openai_result

def generate_1v1_few_shot_prompts(data, args, examples, example_dict):    
    prompts = []
    texts = data.raw_texts
    label_names = [data.label_names[i] for i in args.known_class]
    question = "Is the topic of this paper in the listed categories?"
    for text in texts:
        prompt = "I will first give you examples and you should complete task following the example.\n"
        for num in range(len(examples)):   
            in_context += examples[num] + "\n"
            in_context += "Task: \n"
            if not 'arxiv' in question:
                in_context += "There are following listed categories: \n"
                in_context += "[" + ", ".join(label_names) + "]" + "\n"
                in_context += question + "\n"
            # prompt += exp.topk_confidence(in_context, 3, name)
            prompt += "Question: {}. Provide your best guess and a confidence number. For example,  \
                [{{\"answer\": <your_first_answer>, \"confidence\": <confidence_for_first_answer>}}]        \
                ".format(in_context)
            # prompt += question + "\n"
            prompt += "\nOutput:\n"
            prompt += example_dict[num] + "\n"
        in_context = text + "\n"
        in_context += "Task: \n"
        if not 'arxiv' in question:
            in_context += "There are following categories: \n"
            in_context += "[" + ", ".join(label_names) + "]" + "\n"  
        in_context += question + "\n"
        # prompt += exp.topk_confidence(in_context, 3, name)
        prompt += "Question: {}. Provide your best guess and a confidence number. \
            [{{\"answer\": <your_first_answer>, \"confidence\": <confidence_for_first_answer>}}]        \
            ".format(in_context)
        # prompt += question + "\n"
        prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts 

def generate_1v1_zero_shot_prompts(data, args, major_category_seen, topics_unseen, test_indices, object_cat = "Paper", question = "W?", answer_format = "[{\"answer\":<True or False>, \"confidence\": <confidence_here>, \"category\": <category_here>}]"):
    prompts = []
    texts = [data.raw_texts[i] for i in test_indices]
    label_names = [data.label_fullnames[i] for i in args.known_class]
    
    question = "Is the category of this {} in the category list? Provide your answer and a confidence number between [0-1].".format(object_cat)
    if args.dataset in ["citeseer", "elecomp"]:
        question = "Is the category of this {} in the category list?".format(object_cat) + "\n"
        # answer_format = "[{\"answer:\":<True or False>, \"category\": <category_here>, \"confidence\": <confidence_here>}]"
    question_cons = "Choose False only if you are very certain that the paper does not belong to any of the listed categories."

    question_topic = "If True, specify which category in {} the {} belongs to. If False, provide a suggested category that is not in the category list.".format("[" + ", ".join(label_names) + "]" + "\n", object_cat)

    unseen_topics = "The suggested category includes but not limited to the following: " + "[" + ", ".join(topics_unseen) + ", ...]"
    
    # if args.dataset == "citeseer":
    #     with open('datasets/citeseer_categories.csv', 'r') as f:
    #         label = f.readlines()
    #         label = [x.strip() for x in label][1:
    #         label = [label[i] for i in args.known_class]
    #         label_description = "There are following categories: \n"
    #         label_description += "[" + ", ".join(label_names) + "]"
    #         label_description += " with the following descriptions: \n"
    #         label_description += "[" + ", ".join(label) + "]" + "\n"
    idx = 0
    for text in texts:
        prompt = "The discriprion of {}: \n".format(object_cat)
        prompt += (text + "\n")
        prompt += "Task: \n"
        # if args.dataset == "citeseer":
        #     prompt += label_description
        if not 'arxiv' in question:
            prompt += "There are following categories: \n"
            prompt += "[" + ", ".join(label_names) + "]" + "\n"
        if args.dataset in ["citeseer", "elecomp", 'elephoto']:
            prompt += question + "\n"
            prompt += question_topic + "\n"
            prompt += unseen_topics + "\n"
            # prompt += question_cons + "\n"
            prompt += "Provide your answer, a confidence number between [0-1], and suggested category."
        else:
            prompt += question + "\n"
            prompt += question_cons + "\n"
            prompt += question_topic + "\n"
        
        prompt +=  answer_format
        prompts.append(prompt)
        idx = idx + 1
    return prompts    

def generate_1v1_zero_shot_prompts_nlp(data, label_names, args, major_category_seen, topics_unseen, object_cat = "Paper", question = "W?", answer_format = "[{\"answer\":<True or False>, \"confidence\": <confidence_here>, \"category\": <category_here>}]"):
    prompts = []
    texts = [data[i]['text'] for i in range(len(data))]
    
    question = "Is the topic of this {} in the category list? Provide your answer and a confidence number between [0-1].".format(object_cat)
    if args.dataset in ["news_category"]:
        question = "Is the topic of this paper in the category list?" + "\n"
        # answer_format = "[{\"answer:\":<True or False>, \"category\": <category_here>, \"confidence\": <confidence_here>}]"
    question_cons = "Choose False only if you are very certain that the paper does not belong to any of the listed categories."
    if args.dataset in ["news_category"]:
        question_topic = "If True, specify which category in {} the {} belongs to. If False, provide a suggested category that is not in the category list.".format("[" + ", ".join(label_names) + "]" + "\n", object_cat)
    else:
        question_topic = "If True, specify which category in {} the {} belongs to. If False, provide a suggested category that is not in the category list.".format("[" + ", ".join(label_names) + "]" + "\n", object_cat)
    unseen_topics = "The suggected category includes but not limited to the following: " + "[" + ", ".join(topics_unseen) + ", ...]"
    
    # if args.dataset == "citeseer":
    #     with open('datasets/citeseer_categories.csv', 'r') as f:
    #         label = f.readlines()
    #         label = [x.strip() for x in label][1:]
    #         label = [label[i] for i in args.known_class]
    #         label_description = "There are following categories: \n"
    #         label_description += "[" + ", ".join(label_names) + "]"
    #         label_description += " with the following descriptions: \n"
    #         label_description += "[" + ", ".join(label) + "]" + "\n"
    for text in texts:
        prompt = "{}: \n".format(object_cat)
        prompt += (text + "\n")
        prompt += "Task: \n"
        # if args.dataset == "citeseer":
        #     prompt += label_description
        if not 'arxiv' in question:
            prompt += "There are following categories: \n"
            prompt += "[" + ", ".join(label_names) + "]" + "\n"
        if args.dataset in ["news_category"]:
            prompt += question + "\n"
            prompt += question_topic + "\n"
            prompt += unseen_topics + "\n"
            # prompt += question_cons + "\n"
            prompt += "Provide your answer, a confidence number between [0-1], and suggested category."
        else:
            prompt += question + "\n"
            prompt += question_cons + "\n"
            prompt += question_topic + "\n"
        
        prompt +=  answer_format
        prompts.append(prompt)
    return prompts    


def generate_ood_prompts_nlp(data, args, predicted_category, test_indices_unseen, object_cat = "Paper", question = "W?", answer_format = "[{\"answer:\":<True or False>, \"confidence\": <confidence_here>, \"category\": <category_here>}]"):
    
    prompts = []
    texts = [data[i]["text"] for i in test_indices_unseen]
    
    label_names = predicted_category
    # question = "Which category does this paper belong to? Provide your two guesses as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 1 . For example,  "
    question = "Which category does this {} belong to? Provide your best guess with a confidence ranging from 0 to 1 . For example,  ".format(object_cat)
    answer_format = "[{{\"answer\": <your_first_answer>, \"confidence\": <confidence_for_first_answer>}}]"
    # answer_format = "[{\"answer:\":<category1>, \"confidence\": <confidence1>}, {\"answer:\":<category2>, \"confidence\": <confidence2>}]"
    # question_cons = "Choose False only if you are very certain that the paper does not belong to any of the listed categories."
    # question_topic = "If True, specify which category the paper belongs to. If False, provide a suggested category that is not in the list."
    
    for text in texts:
        prompt = "{}: \n".format(object_cat)
        prompt += (text + "\n")
        prompt += "Task: \n"
        if not 'arxiv' in question:
            prompt += "There are following categories: \n"
            prompt += "[" + ", ".join(label_names) + "]. " + "\n"
        prompt += question + "\n"
        # prompt += question_cons + "\n"
        # prompt += question_topic + "\n"
        
        prompt +=  answer_format
        prompts.append(prompt)
    return prompts        
 
def generate_ood_prompts(data, args, predicted_category, test_indices_unseen, object_cat = "Paper", question = "W?", answer_format = "[{\"answer:\":<True or False>, \"confidence\": <confidence_here>, \"category\": <category_here>}]"):
    
    prompts = []
    texts = [data.raw_texts[i] for i in test_indices_unseen]
    label_names = predicted_category
    # question = "Which category does this paper belong to? Provide your two guesses as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 1 . For example,  "
    question = "Which category does this paper belong to? Provide your best guess with a confidence ranging from 0 to 1 . For example,  "
    answer_format = "[{{\"answer\": <your_first_answer>, \"confidence\": <confidence_for_first_answer>}}]"
    # answer_format = "[{\"answer:\":<category1>, \"confidence\": <confidence1>}, {\"answer:\":<category2>, \"confidence\": <confidence2>}]"
    # question_cons = "Choose False only if you are very certain that the paper does not belong to any of the listed categories."
    # question_topic = "If True, specify which category the paper belongs to. If False, provide a suggested category that is not in the list."
    
    for text in texts:
        prompt = "{}: \n".format(object_cat)
        prompt += (text + "\n")
        prompt += "Task: \n"
        if not 'arxiv' in question:
            prompt += "There are following categories: \n"
            prompt += "[" + ", ".join(label_names) + "]. " + "\n"
        prompt += question + "\n"
        # prompt += question_cons + "\n"
        # prompt += question_topic + "\n"
        
        prompt +=  answer_format
        prompts.append(prompt)
    return prompts           

def get_result_from_output(outputs):
    answer_list = []
    confidence_list = []
    category_list = []
    for i, output in enumerate(outputs):
        output = re.sub(r'answer:', 'answer', output[0][0])
        output = re.sub(r'confidence:', 'confidence', output)
        output = re.sub(r'category:', 'category', output)
        output = re.sub(r'high', '0.9', output)
        output = re.sub(r'low', '0.1', output)
        output = re.sub(r'"True"', 'True', output)
        output = re.sub(r'"False"', 'False', output)
        output = re.sub(r'-', ' ', output)
        # match = re.search(r'"answer":\s*(\w+),\s*"confidence":\s*([\d.]+)', output)
        # match = re.search(r'"answer":\s*(\w+),\s*"confidence":\s*(\d+\.?\d*|"(?:[A-Za-z0-9.]+)")', output)
        match = re.search(r'"answer":\s*(\w+),\s*"confidence":\s*(\d+\.?\d*|"(?:[A-Za-z0-9.]+)"|[A-Za-z]+),\s*"category":\s*(\w+)', output) 
        # match = re.search(r'"answer":\s*(\w+),\s*"category":\s*([\w\s]+),\s*"confidence":\s*([\d.]+)', output)
        if not match:
            # match = re.search(r'"answer":\s*(\w+),\s*"confidence":\s*([\d.]+),\s*"category":\s*"([^"]+)"', output) 
            # match = re.search(r'"answer":\s*"([^"]+)",\s*"confidence":\s*([\d.]+),\s*"category":\s*"([^"]+)"', output)
 
            # match = re.search(r'"answer":\s*(\w+),\s*"category":\s*"([^"]+,\s*"confidence":\s*([\d.]+))"', output) 
            match = re.search(r'"answer":\s*(\w+),\s*"confidence":\s*([\d.]+),\s*"category":\s*"([^"]+)"', output)   
        # match = re.search(r'"answer":\s*(\w+),\s*"confidence":\s*(\d+\.?\d*|"[A-Za-z0-9.]+"),\s*"category":\s*"([\w_]+)"', output)

        if match:
            answer = match.group(1)
            confidence = match.group(2)
            category = match.group(3)
            print(f"Answer: {answer}, Confidence: {confidence}, Category: {category}")
            answer_list.append(answer)
            confidence_list.append(confidence)
            category_list.append(category)
        else:
            try:
                output = ast.literal_eval(output)[0]
                answer = output['answer']
                confidence = output['confidence']
                category = output['category']
                print(f"Answer: {answer}, Confidence: {confidence}, Category: {category}")
                answer_list.append(answer)
                confidence_list.append(confidence)
                category_list.append(category)
            except:            
                print("No match found")
                answer_list.append("")
                confidence_list.append(0)
                category_list.append("")
                print(i)
        # if output[0] == "":
        #     answer.append("")
        #     confidence.append(0)
        # else:
            
        #     answer.append(ast.literal_eval(output[0][0])['answer'])
        #     confidence.append(ast.literal_eval(output[0][0])['confidence'])
    return answer_list, confidence_list, category_list

def get_result_from_ood(outputs):
    answer_list1 = []
    confidence_list1 = []
    answer_list2 = []
    confidence_list2 = []
    
    for i, output in enumerate(outputs):
        output = re.sub(r'answer:', 'answer', output[0][0])
        output = re.sub(r'confidence:', 'confidence', output)
        
        try:      
            output = json.loads(output)
            answer1, confidence1 = output[0]['answer'], output[0]['confidence']
        except:
            answer_match = re.search(r'"answer":\s*"([^"]+)"', output)
            confidence_match = re.search(r'"confidence":\s*([\d.]+)', output)
            
            # 如果匹配到值则返回
            if answer_match and confidence_match:
                answer1 = answer_match.group(1)
                confidence1 = float(confidence_match.group(1))

            else:
                answer1 = ""
                confidence1 = 0
        # answer2, confidence2 = output[1]['answer'], output[1]['confidence']
        
        answer_list1.append(answer1)
        confidence_list1.append(confidence1)
        # answer_list2.append(answer2)
        # confidence_list2.append(confidence2)
        
        print(f"Answer1: {answer1}, Confidence1: {confidence1}")
    
    return answer_list1, confidence_list1
    

def llm_1v1(data, topic, args, major_category_seen, dic_topics_unseen, unseen_indices, seen_indices, test_indices):

    # if test_indices.shape[0] > 2000:
    #     # test_indices = test_indices[:2000]
    #     # np.save('datasets/wikics_test_indices_forllm.npy', test_indices)
    #     if args.dataset == "wikics":
    #         if osp.exists('datasets/wikics_test_indices'+str(args.known_class)+'_forllm.npy'):
    #             test_indices = np.load('datasets/wikics_test_indices'+str(args.known_class)+'_forllm.npy')
    #         else:
    #             test_indices = test_indices[:2000]
    #             np.save('datasets/wikics_test_indices'+str(args.known_class)+'_forllm.npy', test_indices)
    #     if args.dataset == "dblp":
    #         if osp.exists('datasets/dblp_test_indices'+str(args.known_class)+'_forllm.npy'):
    #             test_indices = np.load('datasets/dblp_test_indices'+str(args.known_class)+'_forllm.npy')
    #         else:
    #             test_indices = test_indices[:2000]
    #             np.save('datasets/dblp_test_indices'+str(args.known_class)+'_forllm.npy', test_indices)
    test_indices = test_indices[:1000]
    selected_prompts = generate_1v1_zero_shot_prompts(data, args, major_category_seen, dic_topics_unseen, test_indices, object_cat=topic)
    
    input_filename = "result_1v1/async_input_{}_seen_{}_model_{}_temperature_{}_n_{}_input_seed_{}_method_{}.json".format(args.dataset, str(args.known_class), args.llm_model_method, args.temperature, 1, args.seed, args.prompt_method)
    output_filename = "result_1v1/async_input_{}_seen_{}_model_{}_temperature_{}_n_{}_output_seed_{}_method_{}.json".format(args.dataset, str(args.known_class), args.llm_model_method, args.temperature, 1, args.seed, args.prompt_method)
    # if args.dataset == "cora":
    #     input_filename = "result_1v1/async_input_cora_seen_[1, 3, 4, 5, 6]_model_gpt-4o_temperature_1_n_0.1_input_seed_1_method_100.json"
    #     output_filename = "result_1v1/async_input_cora_seen_[1, 3, 4, 5, 6]_model_gpt-4o_temperature_1_n_0.1_output_seed_1_method_100.json"
    outputs = efficient_openai_text_api(selected_prompts, input_filename, output_filename, model_name=args.llm_model_method, api_key=args.key, temperature=0.1, n=1)
    answer, confidence, category = get_result_from_output(outputs)
    
    return answer, confidence, category, test_indices

def llm_1v1_nlp(data, label_names, args, major_category_seen, dic_topics_unseen, object_cat):
    
    selected_prompts = generate_1v1_zero_shot_prompts_nlp(data, label_names, args, major_category_seen, dic_topics_unseen, object_cat)
    
    input_filename = "result_1v1_nlp/async_input_{}_seen_{}_model_{}_temperature_{}_n_{}_input_seed_{}_method_{}.json".format(args.dataset, str(args.known_class), args.llm_model_method, args.temperature, 1, args.seed, args.prompt_method)
    output_filename = "result_1v1_nlp/async_input_{}_seen_{}_model_{}_temperature_{}_n_{}_output_seed_{}_method_{}.json".format(args.dataset, str(args.known_class), args.llm_model_method, args.temperature, 1, args.seed, args.prompt_method)
    # if args.dataset == "cora":
    #     input_filename = "result_1v1/async_input_cora_seen_[1, 3, 4, 5, 6]_model_gpt-4o_temperature_1_n_0.1_input_seed_1_method_100.json"
    #     output_filename = "result_1v1/async_input_cora_seen_[1, 3, 4, 5, 6]_model_gpt-4o_temperature_1_n_0.1_output_seed_1_method_100.json"
    outputs = efficient_openai_text_api(selected_prompts, input_filename, output_filename, model_name=args.llm_model_method, api_key=args.key, temperature=0.1, n=1)
    answer, confidence, category = get_result_from_output(outputs)
    
    return answer, confidence, category

def llm_llama(model_dir, data, args, seen_indices, unseen_indices, test_indices):
    # model_dir = "/data/projects/punim1970/iclr2025/LLMGNN/Meta-Llama-3-8B"
    # Load tokenizer and model, move model to GPU if available
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.cuda()

    answer_list = []
    confidence_list = []
    category_list = []
    
    # few_shot_topk = configs[args.dataset]['few-shot-seen-unseen']
    fst_example = [data.raw_texts[seen_indices[0]], data.raw_texts[unseen_indices[0]]]
    # fst_result = [few_shot_topk['examples_1'][0][1], few_shot_topk['examples_2'][0][1]]
    full_mapping = load_mapping()
    result1 = '[{{"answer": "True", "confidence": 0.95, "category": "{}"}}]'.format(data.label_fullnames[data.original_y[seen_indices[0]]])
    result2 = '[{{\"answer\": \"False\", \"confidence\": 0.95, \"category\": "{}"}}]'.format(data.label_fullnames[data.original_y[unseen_indices[0]]])   
    fst_result = [result1, result2]
    selected_prompts = few_shot_seen_unseen_prompts(data, args, test_indices, fst_example, fst_result)
    # zero_list = []
    # selected_prompts = generate_1v1_zero_shot_prompts(data, args, zero_list, zero_list, unseen_indices, test_indices)

    
    for i, input_text in enumerate(selected_prompts):
        # Prepare input data and move to GPU
        # inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).cuda()
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: tensor.cuda() for key, tensor in inputs.items()}  # Move each tensor to CUDA

        # Generate text
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1
        )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print the generated text
        print("Generated Text:\n", generated_text)
        
        answer, confidence, category = get_result_from_llamaoutput(generated_text)
        answer_list.append(answer)
        confidence_list.append(confidence)
        category_list.append(category)
    return answer_list, confidence_list, category_list, test_indices     


def llm_llama_new(model_dir, data, args, seen_indices, unseen_indices, test_indices, topics_unseen):
    # model_dir = "/data/projects/punim1970/iclr2025/LLMGNN/Meta-Llama-3-8B"
    # Load tokenizer and model, move model to GPU if available
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.cuda()

    answer_list = []
    confidence_list = []
    category_list = []

    pos_categories = "[" + ", ".join(topics_unseen) + ", ...]"
    selected_prompts = zero_shot_llama_prompts(data, args, test_indices, pos_categories)
    for i, input_text in enumerate(selected_prompts[:2000]):
        # Prepare input data and move to GPU
        # inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).cuda()
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: tensor.cuda() for key, tensor in inputs.items()}  # Move each tensor to CUDA

        # Generate text
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False
        )

        # Decode generated text
        # generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Print the generated text
        print("Generated Text:\n", generated_text)
        
        answer, confidence, category = get_result_from_llamaoutput(generated_text)
        answer_list.append(answer)
        confidence_list.append(confidence)
        category_list.append(category)
    return answer_list, confidence_list, category_list, test_indices[:2000]   

def llm_llama_ood(model_dir, data, args, predicted_category, unseen_indices, test_indices):
    # model_dir = "/data/projects/punim1970/iclr2025/LLMGNN/Meta-Llama-3-8B"
    # Load tokenizer and model, move model to GPU if available
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.cuda()
    answer_list = []
    confidence_list = []
    
    # few_shot_topk = configs[args.dataset]['few-shot-seen-unseen']
    fst_example = [data.raw_texts[unseen_indices[0]]]
    # fst_result = [few_shot_topk['examples_1'][0][1], few_shot_topk['examples_2'][0][1]]
    full_mapping = load_mapping()
    # result1 = '[{{"answer": "True", "confidence": 0.95, "category": "{}"}}]'.format(data.label_fullnames[data.original_y[seen_indices[0]]])
    result = '[{{\"answer\": "{}", \"confidence\": 0.95}}]'.format(data.label_fullnames[data.original_y[unseen_indices[0]]])   
    fst_result = [result]
    selected_prompts = few_shot_ood_prompts(data, args, predicted_category, test_indices, fst_example, fst_result)

    # zero_list = []
    # selected_prompts = generate_1v1_zero_shot_prompts(data, args, zero_list, zero_list, unseen_indices, test_indices)

    
    for i, input_text in enumerate(selected_prompts):
        # Prepare input data and move to GPU
        # inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).cuda()
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: tensor.cuda() for key, tensor in inputs.items()}  # Move each tensor to CUDA

        # Generate text
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1
        )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print the generated text
        print("Generated Text:\n", generated_text)
        
        answer, confidence = get_result_from_llamaoutput_ood(generated_text)
        answer_list.append(answer)
        confidence_list.append(confidence)
    return answer_list, confidence_list


        
def few_shot_seen_unseen_prompts(data, args, test_indices, examples, example_dict, object_cat = "Paper: ", question = "Which arxiv cs subcategories does this paper belong to?", answer_format = " ", name = 'arxiv'):
    prompts = []
    texts = [data.raw_texts[i] for i in test_indices]
    label_names = [data.label_fullnames[i] for i in args.known_class]
    question = "Is the topic of this paper in the listed categories? Provide your answer and a confidence number between [0-1]."
    question_topic = "If True, specify which category in {} the paper belongs to. If False, provide a suggested category that is not in the category list.".format("[" + ", ".join(label_names) + "]" + "\n")
    question_cons = "Choose False only if you are very certain that the paper does not belong to any of the listed categories."
    answer_format = "[{\"answer:\":<True or False>, \"confidence\": <confidence_here>, \"category\": <category_here>}]"
    for text in texts:
        prompt = "I will first give you examples and you should complete task following the example.\n"
        for num in range(len(examples)):            
            in_context = object_cat + examples[num] + "\n"
            in_context += "Task: \n"
            if not 'arxiv' in question:
                in_context += "There are following listed categories: \n"       
                in_context += "[" + ", ".join(label_names) + "]. " + "\n"
                in_context += question + "\n"
                # in_context += question_cons + "\n"
                in_context += question_topic + "\n"
            # prompt += exp.topk_confidence(in_context, 3, name)
            prompt += "Question: {}. For example, {}    \
                ".format(in_context, answer_format)
            # prompt += question + "\n"
            prompt += "\nOutput:\n"
            prompt += example_dict[num] + "\n"
        in_context = text + "\n"
        in_context += "Task: \n"
        if not 'arxiv' in question:
            in_context += "There are following listed categories: \n"
            in_context += "[" + ", ".join(label_names) + "]" + "\n"
            in_context += question + "\n"
            # in_context += question_cons + "\n"
            in_context += question_topic + "\n"

        prompt += "Question: {}. For example, {} \
            ".format(in_context, answer_format)
        # prompt += question + "\n"
        prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts 

# system_role = """
# You are an expert in out-of-distribution identification. Your task is to answer the question.
# """

# prompt = """
# {system_role}
# Here is the question:
# <question>
# {question}
# </question>

# Instructions:
# - If True, specify which category in {categories} the paper belongs to.
# - If False, provide a suggested category that is not in the category list. The suggested category includes but not limited to the following: {pos_categories}.
# - Think through the problem carefully and logically.
# - Be concise and avoid unnecessary information.

# Only output your final answer using this format:
# [
#   {{
#     "answer": "<True or Falsee>",
#     "confidence": <confidence_here>,
#     "category": "<category_here>"
#   }}
# ]

# Your answer:
# """

system_role = """
You are an expert in out-of-distribution (OOD) identification. Your task is to answer the question concisely and accurately.
"""

prompt = """
{system_role}

Question:
<question>
{question}
</question>

Instructions:
- If the paper belongs to one of the provided categories, set "answer" to "True" and put that exact category from {categories} in "category".
- If the paper does NOT belong to any provided category, set "answer" to "False" and provide a suggested category (which may be one of {pos_categories} or a related new label).
- Think carefully and reason logically before answering.
- Be concise and avoid any extra explanation or commentary.

Output requirement:
Return only a single JSON array containing one object, exactly in this format (no extra text):

[
  {{
    "answer": "<True or False>",
    "confidence": <float_between_0_and_1>,
    "category": "<category_name>"
  }}
]

Your answer:
"""


def zero_shot_llama_prompts(data, args, test_indices, pos_categories):
    prompts = []
    texts = [data.raw_texts[i] for i in test_indices]
    label_names = [data.label_fullnames[i] for i in args.known_class]
    question = "Is the topic of this paper in the listed categories? Provide your answer, a confidence number between [0-1], and the category."

    for text in texts:
        in_context = "The discriprion of a paper is: " + text + "\n"
        if not 'arxiv' in question:
            in_context += "There are following listed categories: \n"
            categories = "[" + ", ".join(label_names) + "]" + "\n"
            in_context += categories
            in_context += question + "\n"
            # in_context += question_cons + "\n"
        text_prompt = prompt.format(
                system_role=system_role,
                question=in_context,
                categories=categories,
                pos_categories=pos_categories
            )
        prompts.append(text_prompt)
    return prompts 

def few_shot_ood_prompts(data, args, predicted_category, test_indices, examples, example_dict, object_cat = "Paper", question = "W?", answer_format = "[{\"answer:\":<True or False>, \"confidence\": <confidence_here>, \"category\": <category_here>}]"):
    
    prompts = []
    texts = [data.raw_texts[i] for i in test_indices]
    label_names = predicted_category
    # question = "Which category does this paper belong to? Provide your two guesses as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 1 . For example,  "
    question = "Which category does this paper belong to? Provide your best guess with a confidence ranging from 0 to 1 ."
    answer_format = "[{{\"answer\": <your_first_answer>, \"confidence\": <confidence_for_first_answer>}}]"
    # answer_format = "[{\"answer:\":<category1>, \"confidence\": <confidence1>}, {\"answer:\":<category2>, \"confidence\": <confidence2>}]"
    # question_cons = "Choose False only if you are very certain that the paper does not belong to any of the listed categories."
    # question_topic = "If True, specify which category the paper belongs to. If False, provide a suggested category that is not in the list."
        
    for text in texts:
        prompt = "I will first give you examples and you should complete task following the example.\n"
        for num in range(len(examples)):            
            in_context = object_cat + examples[num] + "\n"
            in_context += "Task: \n"
            if not 'arxiv' in question:
                in_context += "There are following listed categories: \n"       
                in_context += "[" + ", ".join(label_names) + "]. " + "\n"
                in_context += question + "\n"
            # prompt += exp.topk_confidence(in_context, 3, name)
            prompt += "Question: {}. For example, {}    \
                ".format(in_context, answer_format)
            # prompt += question + "\n"
            prompt += "\nOutput:\n"
            prompt += example_dict[num] + "\n"
        in_context = text + "\n"
        in_context += "Task: \n"
        if not 'arxiv' in question:
            in_context += "There are following listed categories: \n"
            in_context += "[" + ", ".join(label_names) + "]" + "\n"
            in_context += question + "\n"

        prompt += "Question: {}. For example, {} \
            ".format(in_context, answer_format)
        # prompt += question + "\n"
        prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts  


def get_result_from_llamaoutput(generated_text):
    # matches = re.findall(r'Your answer:\n(\[.*?\])', generated_text, re.DOTALL)
    matches = re.findall(r'Your answer[^\[]*(\[\s*\{[\s\S]*?\}\s*\])', generated_text)

    if matches:
        # Get the last match
        last_output = matches[-1].strip()

        try:
            # Parse the JSON string
            data = json.loads(last_output)
            
            # Extract the 'answer' and 'confidence' from the first dictionary in the list
            answer = data[0].get("answer")
            answer = re.sub(r'"True"', 'True', answer)
            match = re.search(r'(\d+):', answer)

            if match:
                number = int(match.group(1))  # Convert the extracted number to an integer
                print(f"Extracted number: {number}")
            else:
                print("No number found in the answer string.")
                
            confidence = data[0].get("confidence")
            category = data[0].get("category")
            
            print("Extracted Answer:", answer)
            print("Extracted Confidence:", confidence)
            print("Extracted Category:", category)
            
        except json.JSONDecodeError:
            print("Failed to parse JSON.")
            answer = ""
            confidence = 0
            category = ""
    else:
        print("No Output found in the text.")
    if matches and match:
        return number, confidence, category
    elif matches and not match:
        return answer, confidence, category
    else:
        return None, None, None
    
def get_result_from_llamaoutput_ood(generated_text):
    matches = re.findall(r'Output:\n(\[.*?\])', generated_text, re.DOTALL)

    if matches:
        # Get the last match
        last_output = matches[-1].strip()

        try:
            # Parse the JSON string
            data = json.loads(last_output)
            
            # Extract the 'answer' and 'confidence' from the first dictionary in the list
            answer = data[0].get("answer")               
            confidence = data[0].get("confidence")
            
            print("Extracted Answer:", answer)
            print("Extracted Confidence:", confidence)
            
        except json.JSONDecodeError:
            print("Failed to parse JSON.")
            answer = ""
            confidence = 0
            category = ""
    else:
        print("No Output found in the text.")

    return answer, confidence