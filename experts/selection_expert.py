import numpy as np

from .abstract_expert import Abstract_Expert

class Selection_Expert(Abstract_Expert):
    def __init__(self, name, role = None) -> None:
        super().__init__(name, role)
        
    def select_template(self, jb_template_list):
        # print(f"[Expert] {self.name} selected jailbreak prompts.")
        # randomly select a template
        # print(jb_template_list)
        random_idx = np.random.randint(0, len(jb_template_list))
        selected_template = jb_template_list[random_idx]

        # print(selected_template)
        return selected_template
        
        