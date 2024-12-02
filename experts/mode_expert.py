from .abstract_expert import Abstract_Expert

class Mode_Expert(Abstract_Expert):
    def __init__(self, name, role = None) -> None:
        super().__init__(name, role)
        
    def mode(self, prompt):
        print(f"[Expert] {self.name} mode prompt {prompt}")
        return "Mode"