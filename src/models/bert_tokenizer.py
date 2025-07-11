import torch
from transformers import BertTokenizer
from typing import List, Dict
import random

class BertTextTokenizer:
    def __init__(self, max_length: int = 77, model_name: str = "bert-base-uncased"):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
    
    def encode(self, text: str) -> List[int]:
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        return encoded
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids']
    
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

def generate_cifar10_prompts():
    templates = [
        "a photo of a {cls}",
        "a picture of a {cls}",
        "an image of a {cls}",
        "a {adjective} {cls}",
        "a {adjective} {cls} in {location}",
        "a {color} {cls}",
        "a {size} {cls}",
        "a {cls} {action}",
        "a {quality} photo of a {cls}",
        "a {cls} during {time}"
    ]
    
    class_info = {
        0: {  # airplane
            "cls": ["airplane", "plane", "aircraft", "jet"],
            "adjective": ["flying", "fast", "commercial", "military", "passenger"],
            "location": ["the sky", "clouds", "airport", "runway", "air"],
            "color": ["white", "blue", "silver", "red"],
            "size": ["large", "small", "big", "tiny"],
            "action": ["flying", "landing", "taking off", "in flight"],
            "quality": ["clear", "sharp", "detailed", "bright"],
            "time": ["daytime", "sunset", "morning", "afternoon"]
        },
        1: {  # automobile
            "cls": ["car", "automobile", "vehicle", "sedan"],
            "adjective": ["fast", "modern", "classic", "sleek", "sporty"],
            "location": ["the road", "street", "highway", "parking lot", "city"],
            "color": ["red", "blue", "black", "white", "green"],
            "size": ["small", "large", "compact", "big"],
            "action": ["driving", "parked", "moving", "racing"],
            "quality": ["clear", "detailed", "bright", "sharp"],
            "time": ["daytime", "night", "evening", "morning"]
        },
        2: {  # bird
            "cls": ["bird", "small bird", "songbird", "flying bird"],
            "adjective": ["colorful", "small", "beautiful", "wild", "singing"],
            "location": ["a tree", "the sky", "nature", "branch", "nest"],
            "color": ["blue", "red", "yellow", "colorful", "brown"],
            "size": ["small", "tiny", "little"],
            "action": ["flying", "perched", "singing", "chirping"],
            "quality": ["clear", "natural", "bright", "detailed"],
            "time": ["morning", "daytime", "spring", "summer"]
        },
        3: {  # cat
            "cls": ["cat", "kitten", "feline", "house cat"],
            "adjective": ["cute", "fluffy", "playful", "sleepy", "orange"],
            "location": ["home", "garden", "house", "yard", "indoors"],
            "color": ["orange", "black", "white", "gray", "brown"],
            "size": ["small", "little", "big"],
            "action": ["sleeping", "playing", "sitting", "looking"],
            "quality": ["cute", "adorable", "clear", "detailed"],
            "time": ["daytime", "afternoon", "morning", "evening"]
        },
        4: {  # deer
            "cls": ["deer", "wild deer", "forest deer", "buck"],
            "adjective": ["wild", "graceful", "brown", "beautiful", "young"],
            "location": ["forest", "woods", "nature", "field", "wilderness"],
            "color": ["brown", "tan", "natural"],
            "size": ["large", "medium", "big"],
            "action": ["grazing", "standing", "running", "looking"],
            "quality": ["natural", "wild", "clear", "detailed"],
            "time": ["morning", "daytime", "autumn", "spring"]
        },
        5: {  # dog
            "cls": ["dog", "puppy", "pet dog", "canine"],
            "adjective": ["happy", "playful", "friendly", "cute", "loyal"],
            "location": ["yard", "park", "home", "garden", "outdoors"],
            "color": ["brown", "black", "white", "golden", "tan"],
            "size": ["small", "large", "medium", "big"],
            "action": ["playing", "running", "sitting", "looking"],
            "quality": ["cute", "happy", "clear", "detailed"],
            "time": ["daytime", "morning", "afternoon", "sunny day"]
        },
        6: {  # frog
            "cls": ["frog", "small frog", "green frog", "tree frog"],
            "adjective": ["green", "small", "wet", "jumping", "amphibian"],
            "location": ["pond", "water", "lily pad", "swamp", "nature"],
            "color": ["green", "brown", "dark green"],
            "size": ["small", "tiny", "little"],
            "action": ["jumping", "sitting", "swimming", "croaking"],
            "quality": ["clear", "natural", "detailed", "bright"],
            "time": ["daytime", "morning", "spring", "summer"]
        },
        7: {  # horse
            "cls": ["horse", "wild horse", "stallion", "mare"],
            "adjective": ["brown", "beautiful", "wild", "strong", "graceful"],
            "location": ["field", "pasture", "farm", "meadow", "countryside"],
            "color": ["brown", "black", "white", "chestnut", "bay"],
            "size": ["large", "big", "majestic"],
            "action": ["running", "galloping", "standing", "grazing"],
            "quality": ["majestic", "beautiful", "clear", "detailed"],
            "time": ["daytime", "morning", "afternoon", "sunset"]
        },
        8: {  # ship
            "cls": ["ship", "boat", "vessel", "sailing ship"],
            "adjective": ["large", "white", "sailing", "ocean", "cargo"],
            "location": ["ocean", "sea", "water", "harbor", "port"],
            "color": ["white", "blue", "red", "gray"],
            "size": ["large", "big", "huge", "massive"],
            "action": ["sailing", "floating", "moving", "docked"],
            "quality": ["clear", "detailed", "bright", "sharp"],
            "time": ["daytime", "morning", "afternoon", "sunset"]
        },
        9: {  # truck
            "cls": ["truck", "large truck", "cargo truck", "delivery truck"],
            "adjective": ["large", "heavy", "cargo", "delivery", "commercial"],
            "location": ["road", "highway", "street", "city", "loading dock"],
            "color": ["red", "blue", "white", "black", "yellow"],
            "size": ["large", "big", "huge", "massive"],
            "action": ["driving", "moving", "parked", "loading"],
            "quality": ["clear", "detailed", "bright", "sharp"],
            "time": ["daytime", "morning", "afternoon", "evening"]
        }
    }
    
    return templates, class_info

def get_prompts(dataset_name):
    if dataset_name == "mnist":
        return {
            0: ["handwritten digit zero", "the number 0", "zero digit", "handwritten zero"],
            1: ["handwritten digit one", "the number 1", "one digit", "handwritten one"],
            2: ["handwritten digit two", "the number 2", "two digit", "handwritten two"],
            3: ["handwritten digit three", "the number 3", "three digit", "handwritten three"],
            4: ["handwritten digit four", "the number 4", "four digit", "handwritten four"],
            5: ["handwritten digit five", "the number 5", "five digit", "handwritten five"],
            6: ["handwritten digit six", "the number 6", "six digit", "handwritten six"],
            7: ["handwritten digit seven", "the number 7", "seven digit", "handwritten seven"],
            8: ["handwritten digit eight", "the number 8", "eight digit", "handwritten eight"],
            9: ["handwritten digit nine", "the number 9", "nine digit", "handwritten nine"]
        }
    elif dataset_name == "cifar10":
        templates, class_info = generate_cifar10_prompts()
        prompts = {}
        
        for class_id, info in class_info.items():
            class_prompts = []
            
            for _ in range(20):  # Generate 20 prompts per class
                template = random.choice(templates)
                
                prompt = template.format(
                    cls=random.choice(info["cls"]),
                    adjective=random.choice(info["adjective"]),
                    location=random.choice(info["location"]),
                    color=random.choice(info["color"]),
                    size=random.choice(info["size"]),
                    action=random.choice(info["action"]),
                    quality=random.choice(info["quality"]),
                    time=random.choice(info["time"])
                )
                
                class_prompts.append(prompt)
            
            prompts[class_id] = class_prompts
        
        return prompts
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def test_tokenizer():
    tokenizer = BertTextTokenizer()
    
    prompts = get_prompts("cifar10")
    
    print("Testing BERT tokenizer with generated CIFAR-10 prompts:")
    for class_id in range(3):  # Test first 3 classes
        print(f"\nClass {class_id} prompts:")
        for i, prompt in enumerate(prompts[class_id][:5]):  # Show first 5 prompts
            tokens = tokenizer.encode(prompt)
            print(f"  {i+1}. {prompt}")
    
    print(f"\nVocab size: {tokenizer.vocab_size}")

if __name__ == "__main__":
    test_tokenizer()