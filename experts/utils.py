import threading
import copy
from argparse import Namespace
import torch
import json
from termcolor import colored, cprint
import pandas as pd
import chromadb
import time
from typing import Any, TypedDict,List, Optional
import ast
from together import Together
import openai

def get_completion(messages, model="gpt-4o-mini", temperature=0.1):
    # messages = [{"role":"system", "content" : sys_prompt}, {"role":"user", "content" : prompt}]
    response = ''
    except_waiting_time = 0.1
    while response == '':
        try:
            # response = openai.ChatCompletion.create(
            #     model=model,
            #     messages=messages,
            #     temperature=temperature,
            #     request_timeout=50
            # )
            with openai.OpenAI() as client:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    # request_timeout=50
                )
            # k_tokens = response["usage"]["total_tokens"]/1000
            # p_tokens = response["usage"]["prompt_tokens"]/1000
            # r_tokens = response["usage"]["completion_tokens"]/1000

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(except_waiting_time)
            if except_waiting_time < 2:
                except_waiting_time *= 2
    return response.choices[0].message.content

def get_completion_gpt(messages, model="gpt-4o-mini", temperature=0.1):
    # messages = [{"role":"system", "content" : sys_prompt}, {"role":"user", "content" : prompt}]
    response = ''
    except_waiting_time = 0.1
    while response == '':
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                request_timeout=50
            )
            k_tokens = response["usage"]["total_tokens"]/1000
            p_tokens = response["usage"]["prompt_tokens"]/1000
            r_tokens = response["usage"]["completion_tokens"]/1000

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(except_waiting_time)
            if except_waiting_time < 2:
                except_waiting_time *= 2
    return response.choices[0].message["content"]