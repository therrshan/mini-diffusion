import torch
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self, max_length: int = 77):
        self.max_length = max_length
        
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<start>": 2,
            "<end>": 3
        }
        
        self.vocab = {
            "handwritten": 4,
            "digit": 5,
            "zero": 6,
            "one": 7,
            "two": 8,
            "three": 9,
            "four": 10,
            "five": 11,
            "six": 12,
            "seven": 13,
            "eight": 14,
            "nine": 15,
            "number": 16,
            "0": 17,
            "1": 18,
            "2": 19,
            "3": 20,
            "4": 21,
            "5": 22,
            "6": 23,
            "7": 24,
            "8": 25,
            "9": 26
        }
        
        self.vocab.update(self.special_tokens)
        self.vocab_size = len(self.vocab)
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        tokens = text.split()
        return tokens
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        
        token_ids = [self.special_tokens["<start>"]]
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.special_tokens["<unk>"])
        
        token_ids.append(self.special_tokens["<end>"])
        
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.special_tokens["<pad>"]] * (self.max_length - len(token_ids)))
        
        return token_ids
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        batch_ids = []
        for text in texts:
            token_ids = self.encode(text)
            batch_ids.append(token_ids)
        
        return torch.tensor(batch_ids, dtype=torch.long)
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ["<pad>", "<start>", "<end>"]:
                    tokens.append(token)
        
        return " ".join(tokens)

def get_digit_prompts():
    prompts = {
        0: ["handwritten digit zero", "digit 0", "zero"],
        1: ["handwritten digit one", "digit 1", "one"],
        2: ["handwritten digit two", "digit 2", "two"],
        3: ["handwritten digit three", "digit 3", "three"],
        4: ["handwritten digit four", "digit 4", "four"],
        5: ["handwritten digit five", "digit 5", "five"],
        6: ["handwritten digit six", "digit 6", "six"],
        7: ["handwritten digit seven", "digit 7", "seven"],
        8: ["handwritten digit eight", "digit 8", "eight"],
        9: ["handwritten digit nine", "digit 9", "nine"]
    }
    return prompts