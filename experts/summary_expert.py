from .abstract_expert import Abstract_Expert 
import pandas as pd
from .utils import get_completion
from termcolor import colored, cprint

class Summary_Expert(Abstract_Expert): 
    def __init__(self, name, role = None) -> None:
        super().__init__(name, role)
        

    def summarize_and_init_memory(self, previous_file):
        cprint(f"[Expert] {self.name} is summarizing the previous results from {previous_file}", 'yellow')
        success_memory = []
        df = pd.read_csv(previous_file)
        df_refusal = df.query("refusal == 0")
        for line in df_refusal.values:
            success_memory.append(line[2] + line[3])
        # print(success_memory)
        
        return success_memory
        
        
