import argparse
import json
import logging
import os
import pickle
import re
import time
from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError, APIError, APIConnectionError
from tqdm import tqdm 
import multiprocessing as mp
from prompt_selection import Demon_sampler

class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name, max_tokens, api_key):
        self.args = args
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.token_num = 0
    
    def get_response(self, input_text, turn_type):
        if self.args.debug:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            print("query API to get message:\n%s" % message['content'])
            response = input("input the returned response:")
        else:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message.content)
            response = message.content.strip()
        return response

    def query_localLLM_to_get_response(self,message):
        # input: message: {role': 'user', 'content': string(input_text to LLM, which has implemented) }
        # return:  response: {role': 'assistant', 'content': string(output_text wich need you to fetch and store here)}
        output_text = "" #modifiy here
        response = {'role': 'assistant', 'content': output_text}
        if output_text == "":
            print("Implement The function")
        return response
    
    def create_message(self, input_text, turn_type):
        if turn_type == "init_query":  
            template = self.prompt['init_query']
            demonstrations_for_r, r_text = input_text
            input_text =template.format(demonstrations_for_r =demonstrations_for_r, r_text = r_text, r_text2 = r_text)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages):
        while True:
            try:
                res = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                if args.debug_online:
                    print(res)
                self.token_num = res.usage.total_tokens
                return res.choices[0].message
            except RateLimitError:
                print('openai.RateLimitError\nRetrying...')
                time.sleep(30)
            except APIStatusError:
                print('openai.APIStatusError\nRetrying...')
                time.sleep(20)
            except APITimeoutError:
                print('openai.APITimeoutError\nRetrying...')
                time.sleep(20)
            except APIError:
                print('openai.APIError\nRetrying...')
                time.sleep(20)
            except APIConnectionError:
                print('openai.APIConnectionError\nRetrying...')
                time.sleep(20)


    def reset_history(self):
        self.history_messages = []
        self.history_contents = []
        self.token_num = 0
        
    def reset_history_messages(self):
        self.history_messages = []

    def reset_history_contents(self):
        self.history_contents = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]
        
        
from collections import defaultdict
class Solver:
    def __init__(self, args, api_key):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name,
                           max_tokens=args.max_tokens, api_key=api_key)
        
        self.log = []
        
        self.ent2text = defaultdict(str)
        self.rel2text = defaultdict(str)
        self.load_ent_to_text()
        self.load_rel_to_text()
                
    def forward(self, r, triples): #Here tpe_id not a int id, but like '/m/08966'
        self.LLM.reset_history()
        self.reset_history()
        demonstration_text = self.serialize_demonstrations(triples)
        r_text = self.relation_text(r)
        final_response = self.LLM.get_response((demonstration_text, r_text),"init_query")
        self.log.append(final_response)
        
        return final_response, self.LLM.history_contents
        
        



    def relation_text(self,relation):
        if args.dataset == 'wn18rr':
            return self.rel2text[relation]
        else:
            relation_hierachy_list = relation.strip().replace('.',' ').split('_')
            final_string = ''
            for st in reversed(relation_hierachy_list): 
                if st != "":
                    final_string += st + " of "
            return final_string
    
    def serialize_demonstrations(self,demon_triples):
        demon_text = ""
        for tp in demon_triples:
            demon_text += self.generate_demonstration_text(tp) + '\n'
        demon_text.strip()
        if demon_text == "": demon_text = "None."
        return demon_text
        
    def generate_demonstration_text(self, triple):
        h,r,t = triple
        h = self.ent2text[h]
        t = self.ent2text[t]
        if args.dataset == 'wn18rr': demonstration_text = t + " " + self.relation_text(r) + " " + h
        else: demonstration_text = t + " is the " + self.relation_text(r) + h
        return demonstration_text
    
    def reset_history(self):
        self.log = []
    
    def load_ent_to_text(self):
        with open('dataset/' + args.dataset + '/entity2text.txt', 'r', encoding='utf-8') as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text
    def load_rel_to_text(self):
        with open('dataset/' + args.dataset + '/relation2text.txt', 'r', encoding='utf-8') as file:
            rel_lines = file.readlines()
            for line in rel_lines:
                rel, text = line.strip().split("\t")
                self.rel2text[rel] = text
                    
    
def main(args, demonstration_r, idx, api_key):
    from collections import defaultdict
    

    if idx == -1:
        output_path = args.output_path
        chat_log_path = args.chat_log_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_path = args.output_path + "_" + idx
        chat_log_path = args.chat_log_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), chat_log_path))
    solver = Solver(args, api_key)

    import random
    total_items = len(demonstration_r)
    with open(output_path, "w", encoding='utf-8') as f:
        with open(chat_log_path, "w", encoding='utf-8') as fclog:
            for i, key in enumerate(demonstration_r):
                print(f"--- PID {os.getpid()}: Processing item {i+1}/{total_items}, Relation: {key} ---")
                try:
                    r = key
                    triples = random.sample(demonstration_r[r],args.demon_per_r)        
                    clean_relation, chat_history = solver.forward(r, triples)
                except Exception as e:
                    print(e)
                    continue

                clean_text = defaultdict(str)
                clean_text["Raw"] = key
                clean_text["Description"] = clean_relation
                f.write(json.dumps(clean_text) + "\n")
                
                chat = str(key) + "\n" + "\n******\n".join(chat_history) + "\n------------------------------------------\n"
                fclog.write(chat)

    print("---------------PID %d end--------------" % (os.getpid()))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wn18rr")
    parser.add_argument('--output_path', default="./dataset/wn18rr/alignment/alignment_output.txt")
    parser.add_argument('--chat_log_path', default="./dataset/wn18rr/alignment/alignment_chat.txt")


    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--debug_online', action="store_true")

    parser.add_argument('--max_tokens', default=300, type=int, help='max-token')
    parser.add_argument('--prompt_path', default="./prompts/text_alignment.json")
    parser.add_argument('--prompt_name', default="chat", )
    parser.add_argument('--bagging_type', default="llm", )

    parser.add_argument('--device', default=0, help='the gpu device')
    
    parser.add_argument('--api_key', default="", type=str)
    parser.add_argument('--demon_per_r', default=30)
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')


    args = parser.parse_args()

    print("Start querying the LLM.")
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r", encoding='utf-8') as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)
    
    demonstration_r =defaultdict(list)
    with open("dataset/" + args.dataset + "/demonstration/all_r_triples.txt", "r", encoding='utf-8') as f:
        demonstration_r = json.load(f)

    if args.debug_online:
        import itertools
        num_test_items = 4 
        demonstration_r = dict(itertools.islice(demonstration_r.items(), num_test_items))
        print(f"--- Running in debug_online mode: processing only {len(demonstration_r)} items. ---")

    if args.num_process == 1:
        main(args, demonstration_r, idx=-1, api_key=args.api_key)
    else:
        pass