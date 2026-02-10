#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/openclaw.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs('logs', exist_ok=True)
os.makedirs('memory', exist_ok=True)
os.makedirs('models', exist_ok=True)

class OpenClaw:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.history: List[Dict] = []
        self.max_history = 5
        self.model_path = "./models/llama-3.2-1b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        if self.model is not None:
            return
            
        logger.info("Loading Llama 3.2 1B (4-bit)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            self._download_and_load()
    
    def _download_and_load(self):
        model_id = "unsloth/Llama-3.2-1B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info("Model downloaded and cached")
    
    def chat(self, user_input: str) -> str:
        if self.model is None:
            self.load_model()
        
        system_msg = "You are OpenClaw, a helpful AI assistant. Be concise and accurate."
        
        messages = [{"role": "system", "content": system_msg}]
        
        for exchange in self.history[-self.max_history:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["bot"]})
        
        messages.append({"role": "user", "content": user_input})
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        self.history.append({
            "time": datetime.now().isoformat(),
            "user": user_input,
            "bot": response
        })
        
        with open('memory/history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return response.strip()
    
    def load_history(self):
        try:
            with open('memory/history.json', 'r') as f:
                self.history = json.load(f)
            logger.info(f"Loaded {len(self.history)} previous messages")
        except FileNotFoundError:
            pass

def main():
    print("\n" + "="*50)
    print("🤖 OpenClaw (Llama 3.2 1B)")
    print("Type 'quit' to exit, 'clear' to reset chat")
    print("="*50 + "\n")
    
    bot = OpenClaw()
    bot.load_history()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                bot.history = []
                print("Chat cleared.")
                continue
            elif not user_input:
                continue
            
            print("Bot: ", end="", flush=True)
            response = bot.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
