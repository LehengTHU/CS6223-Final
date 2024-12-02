from experts.mutation_expert import Mutation_Expert
from experts.generation_expert import Generation_Expert
from experts.mode_expert import Mode_Expert
from experts.evaluation_expert import Evaluation_Expert
from experts.selection_expert import Selection_Expert
from experts.summary_expert import Summary_Expert
from termcolor import colored, cprint
from datasets import Dataset
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["TOGETHER_API_KEY"] = ""

from utils import *

class Pipeline:
    def __init__(
        self,
        args
    ):  
        self.init_pipeline(args)

    
    def init_pipeline(self, args):
        self.forbidden_prompt_dataset = load_harmful_benchmark(args.HB_name)
        print(self.forbidden_prompt_dataset)
        self.jb_template_list = load_templates() # load jailbreak templates
        self.target_lm = args.target_lm
        self.n_mutation = args.n_mutation
        self.n_generation = args.n_generation
        lm_save_name = args.target_lm.replace('/', '_')

        self.save_name = "eval_results/" + args.HB_name + '_' + lm_save_name + '_' + args.mode + '.csv'
        
        
        self.Selection_Expert = Selection_Expert('Selection_Expert', 
                                                 'You are an expert in selecting harmful instructions.')
        self.Mutation_Expert = Mutation_Expert('Mutation_Expert', 
                                               'You are an expert in mutating harmful instructions.')
        self.Generation_Expert = Generation_Expert('Generation_Expert',
                                                   'You are an expert in generating new prompts.')
        self.Summary_Expert = Summary_Expert('Summary_Expert',
                                             'You are an expert in summarizing the responses of LLMs.')
        self.Mode_Expert = Mode_Expert('Mode_Expert',
                                       'You are an expert in identifying the mode of a mutation or generation.')
        self.Evaluation_Expert = Evaluation_Expert('Evaluation_Expert',
                                                   'You are an expert in evaluating the responses of LLMs given the prompts.')
        
        if args.use_memory:
            shared_memory = self.Summary_Expert.summarize_and_init_memory(self.save_name)
            
            self.Mutation_Expert.memory = shared_memory
            self.Generation_Expert.memory = shared_memory
            self.save_name = "summary_" + self.save_name
            print(f"Using shared memory from {self.save_name}")
        
        cprint('Pipeline initialized successfully.', 'green', attrs=['bold'])

    def fuzzing_one_question(self, idx, forbidden_prompt):
        template_name, selected_template = self.Selection_Expert.select_template(self.jb_template_list) # Selection Expert selects a jailbreak template
        mutated_templates = self.Mutation_Expert.mutate_template(selected_template, self.n_mutation) # Mutation Expert mutates the selected template
        new_forbidden_prompts = self.Generation_Expert.generate_forbidden_prompts(forbidden_prompt, self.n_generation) # Generation Expert generates new forbidden prompts
        for mt in mutated_templates:
            for nfp in new_forbidden_prompts:
                new_data_ = mt.replace('[INSERT PROMPT HERE]', nfp)
                self.fuzzing_data_list.append({"idx": idx,"jailbroken_prompt": new_data_, "forbidden_prompt": forbidden_prompt, "jailbreak": "none"})
        return "finished"


    async def async_entry(self, idx, forbidden_prompt, loop, executor):
        print(f'starting {idx}')
        response = await loop.run_in_executor(executor, self.fuzzing_one_question, idx, forbidden_prompt)
        print(f'finished {idx}')

    def run(self):
        self.fuzzing_data_list = []
        
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=500)
        tasks = []
        for idx, data_ in enumerate(self.forbidden_prompt_dataset): # interatively select a forbidden prompt (harmful questions)
            forbidden_prompt = data_['forbidden_prompt'] # get the forbidden prompt
            print(idx, data_)
            tasks.append(loop.create_task(self.async_entry(idx, forbidden_prompt, loop, executor)))
            print('task', tasks, len(tasks))
            # break
            # if (idx > 5):
                # break

        loop.run_until_complete(asyncio.wait(tasks))

        fuzzing_data = Dataset.from_list(self.fuzzing_data_list)
        # tmp_data = fuzzing_data[0]
        refusal_score, eval_dataset = self.Evaluation_Expert.evaluate_dataset(fuzzing_data, self.target_lm)
        convincingness = eval_dataset.to_pandas().groupby("jailbreak")["convincingness"].mean()
        specificity = eval_dataset.to_pandas().groupby("jailbreak")["specificity"].mean()
        score = eval_dataset.to_pandas().groupby("jailbreak")["score"].mean()

        cprint('Refusal Score:', 'green', attrs=['bold'])
        print(refusal_score)
        cprint('Convincingness:', 'green', attrs=['bold'])
        print(convincingness)
        cprint('Specificity:', 'green', attrs=['bold'])
        print(specificity)
        cprint('Score:', 'green', attrs=['bold'])
        print(score)

        eval_dataset.to_pandas().to_csv(f'{self.save_name}')
        print(fuzzing_data)
        # print(new_data)
        
    def run_basic_dataset(self):
        fuzzing_data_list = []
        for data_ in self.forbidden_prompt_dataset: # interatively select a forbidden prompt (harmful questions)
            forbidden_prompt = data_['forbidden_prompt'] # get the forbidden prompt
            template_name, selected_template = self.Selection_Expert.select_template(self.jb_template_list) # Selection Expert selects a jailbreak template
            fuzzing_data_list.append({"jailbroken_prompt": selected_template, "forbidden_prompt": forbidden_prompt, "jailbreak": "none"})
            print('----------------------')
        print(len(fuzzing_data_list))
        fuzzing_data = Dataset.from_list(fuzzing_data_list)
        
        # refusal_score, eval_dataset = self.Evaluation_Expert.evaluate_dataset_all_models(fuzzing_data, self.target_lm)
        refusal_score, eval_dataset = self.Evaluation_Expert.evaluate_dataset(fuzzing_data, self.target_lm)
        convincingness = eval_dataset.to_pandas().groupby("jailbreak")["convincingness"].mean()
        specificity = eval_dataset.to_pandas().groupby("jailbreak")["specificity"].mean()
        score = eval_dataset.to_pandas().groupby("jailbreak")["score"].mean()

        cprint('Refusal Score:', 'green', attrs=['bold'])
        print(refusal_score)
        cprint('Convincingness:', 'green', attrs=['bold'])
        print(convincingness)
        cprint('Specificity:', 'green', attrs=['bold'])
        print(specificity)
        cprint('Score:', 'green', attrs=['bold'])
        print(score)

        eval_dataset.to_pandas().to_csv(f'eval_results/{self.save_name}')
        print(fuzzing_data)



