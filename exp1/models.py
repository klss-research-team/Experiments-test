# models.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from exp1.prompts import parse_json_output

class LLMPolicy:
    '''
    Wrapper for a HuggingFace causal language model.
    Used for auction agents and evaluator LLM.
    '''

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(self, prompt, max_new_tokens=256, temperature=0.7):
        '''
        Generates raw text from the LLM.
        '''
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # remove prompt from output if model repeats it
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        return generated_text.strip()

    def generate_json(self, prompt, max_new_tokens=128, temperature=0.7, max_retries=3):
        '''
        Generates text and tries to parse it as JSON
        '''
        for _ in range(max_retries):

            raw_output = self.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            parsed = parse_json_output(raw_output)

            if parsed is not None:
                return parsed, raw_output

        return None, raw_output