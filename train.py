from src.utils.training import *
from src.scripts.load_and_save import * 


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
model, data_collator = generate_model(tokenizer, max_in_size=512, multiplier=1)
tokenized_dataset = Dataset.load_from_disk("data/tokenized_dataset")
model, tokenizer, logs = train_fundation(model, tokenizer, data_collator, tokenized_dataset)