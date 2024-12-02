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
from .utils import *

class Abstract_Expert:
    # Initialize the Expert with a name and a role
    def __init__(self, name, role = None) -> None:
        super().__init__()
        self.name = name
        self.role = role
        print(f"[Expert] initialized with name {name}")
        self.init_role()
        self.init_memory()
    
    # Initialize the role of the Expert
    def init_role(self):
        self.role_prompt = self.role
        print(f"[Expert] {self.name} is initialized with role")
    
    # Initialize the memory of the Expert
    def init_memory(self):
        self.memory_client = chromadb.Client()
        # switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
        self.memory = self.memory_client.get_or_create_collection(name=self.name)
        print(f"[Expert] {self.name} is initialized with memory")
        # switch `add` to `upsert` to avoid adding the same documents every time
    
    # Print all the memory of the Expert
    def print_memory(self):
        print(f"Expert {self.name} has memory:")
        print(self.memory.get())
    
    # Add a memory to the Expert
    def add_memory(self, memory_name, memory_content):
        self.memory.upsert(
            documents=[
                memory_content
            ],
            ids=[memory_name]
        )
        print(f"[Expert] {self.name} added memory {memory_name}: {memory_content}")
    
    # Query the memory of the Expert
    def query_memory(self, query, n_ret = 1):
        results = self.memory.query(
            query_texts=[query], # Chroma will embed this for you
            n_results=n_ret # how many results to return
        )
        print(f"[Expert] {self.name} queried memory with '{query}'")
        return results
    
    
    def extract_key_content(self, query, sentence):
        messages = [{"role":"system", "content" : f"Extract the content most related to the user query {query} in this sentence, and output it. Do not generate any other unrelated words."}, 
                    {"role":"user", "content" : f"The sentence is '{sentence}'"}]
        key_content = get_completion(messages)
        return key_content
    
    def generate_response(self, request):
        # print(f"[Expert] {self.name} is generating response to user request")
        # return "I am a general Expert"
        # print('!!!!', self.role, request)
        message = [{"role":"system", "content" : self.role}, {"role":"user", "content" : request}]
        # print('!!!!', message)
        return get_completion(message)

    def generate_response_gpt(self, request):
        # print(f"[Expert] {self.name} is generating response to user request")
        # return "I am a general Expert"
        # print('!!!!', self.role, request)
        message = [{"role":"system", "content" : self.role}, {"role":"user", "content" : request}]
        # print('!!!!', message)
        return get_completion_gpt(message)