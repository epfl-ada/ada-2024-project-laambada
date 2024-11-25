from collections import Counter
from itertools import chain
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import re
import torch
from torch import nn
from trl import SFTTrainer
from datasets import Dataset
from sklearn.preprocessing import OrdinalEncoder
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
import tokenizers.decoders as decoders
import tokenizers.processors as processors
from tokenizers.pre_tokenizers import Whitespace, PreTokenizer
from tokenizers.trainers import WordPieceTrainer
from transformers import (
    MambaConfig,
    MambaForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
)

def atom_set(smiles_list):
    """
    Extract the set of atom symbols from a list of SMILES strings.
    """
    ligand_atoms = set()
    ligands = set(smiles_list)  # Deduplicate ligands

    for ligand in tqdm(ligands):
        ligand_atoms.update(filter(str.isalpha, ligand))
    return ligand_atoms

def process_ngram(n, ligands):
    grams = Counter()
    for ligand in ligands:
        for i in range(len(ligand) - n + 1):
            grams[ligand[i:i+n]] += 1
    return grams.most_common(10)

def parallel_ngrams(data):
    ligands = set(data["ligand"].tolist())
    
    # Create a pool of workers
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=min(num_cores, 5))  # Use at most 5 processes (one per n-gram)
    
    # Prepare the worker function with fixed ligands argument
    worker = partial(process_ngram, ligands=ligands)
    
    # Process n-grams in parallel
    results = list(tqdm(pool.imap(worker, range(1, 6)), total=5))
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Combine results
    all_common_grams = list(chain.from_iterable(results))
    
    # Extract just the grams without frequencies
    common_grams = [gram for gram, freq in all_common_grams]
    
    # Print results
    for n, result in enumerate(results, 1):
        print(f"{n}-grams:", result)
    print("\nAll common grams:", common_grams)
    
    return common_grams

def build_vocab(common_grams={""}, ligand_atoms={""}):
    total_vocab = set(common_grams).union(ligand_atoms)
    total_vocab.update([
        '=', '#', '-', '+', '(', ')', '[', ']', '/', '\\', '@', '%', '.', ':', '*',  # Special characters
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Numbers
        ' '
    ])
    return list(total_vocab)

class SMILEPreTokenizer(PreTokenizer):
    def __init__(self, total_vocab):
        super.__init__()
        self.total_vocab = total_vocab

    def split(self, text):
        # Regex to split SMILES into meaningful tokens
        escaped_tokens = [re.escape(token) for token in self.total_vocab]
        pattern = re.compile('|'.join(sorted(escaped_tokens, key=len, reverse=True)))
        tokens = re.findall(pattern, text)
        # Return tokens with positions (offsets are not accurate here)
        offset = 0
        tokens_with_offsets = []
        for token in tokens:
            start = text.find(token, offset)
            end = start + len(token)
            tokens_with_offsets.append((token, (start, end)))
            offset = end
        return tokens_with_offsets
    
    def pre_tokenize(self, pretok):
        pretok.split(self.split)

def generate_tokenizer(df, pre_tokenizer=None):
    ligands = df['ligand'].tolist()
    # Initialize the tokenizer with a WordPiece model
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizer if pre_tokenizer else Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    # Define a trainer for the tokenizer
    trainer = WordPieceTrainer(
        vocab_size=100,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    # Load the dataset
    dataset = Dataset.from_list([{ "text": ligand } for ligand in ligands])

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i+batch_size]["text"]
            
    # Train the tokenizer on the dataset
    tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ligands))
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)   
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

def generate_model(tokenizer, max_in_size=64, multiplier=1):
    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)
    config = MambaConfig(
        vocab_size=vocab_size,
        intermediate_size=256,  # Reduced from default
        max_position_embeddings=max_in_size,
        num_hidden_layers=1*multiplier,    # Reduced from default
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = MambaForCausalLM(config)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    return model, data_collator

def tokenize_smiles_dataset(df, tokenizer, max_length=64):
    tokenized_data = []
    unique_ligands = set(df['ligand'].tolist())
    for ligand in tqdm(unique_ligands, desc="Tokenizing SMILES"):
        tokenized_data.append(tokenizer(ligand, padding=True, truncation=True, max_length=max_length))
        
    tokenized_dataset = Dataset.from_dict({
        **{key: [d[key] for d in tokenized_data] for key in tokenized_data[0]},
    })
    # Add labels column with same value as input_ids
    tokenized_dataset = tokenized_dataset.add_column(
        "labels", tokenized_dataset["input_ids"].copy()
    )
    
    tokenized_dataset.save_to_disk("data/tokenized_dataset")
    return tokenized_dataset

def train_fundation(model, tokenizer, data_collator, tokenized_dataset): 
    # Disable WANDB logging
    import os
    os.environ["WANDB_DISABLED"] = "true"   
    training_args = TrainingArguments(
        output_dir="./mamba-smiles",
        save_strategy="steps",
        save_steps=1000,
        learning_rate=5e-4,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 5e-6},
        weight_decay=0.1,
        num_train_epochs=1,
        save_total_limit=6,
        logging_steps=10,
        per_device_train_batch_size=512,
        gradient_accumulation_steps=1,
        bf16=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
    )

    training_dataset = tokenized_dataset
    
    # Initialize Trainer
    trainer = SFTTrainer(
        model=model.to("cuda"),
        args=training_args,
        train_dataset=training_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        packing=True,
        max_seq_length=model.config.max_position_embeddings,
    )

    # Train the model
    logs = trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./model/mamba-smiles-base")
    tokenizer.save_pretrained("./model/mamba-smiles-base")

    return model, tokenizer, logs

class IC50DataCollator:
    def __call__(self, features):
        batch = {
            'ligand_embedding': torch.stack([torch.tensor(f['ligand_embedding']) for f in features]),
            'protein_index': torch.tensor([f['protein_index'] for f in features], dtype=torch.int64),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.float)
        }
        return batch

class HFIC50Predictor(nn.Module):
    def __init__(self, embedding_size, num_proteins):
        super(HFIC50Predictor, self).__init__()
        self.protein_embedding = nn.Embedding(num_proteins, embedding_size)
        self.fc = nn.Sequential(
            nn.Linear(embedding_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ligand_embedding, protein_index, labels=None):
        protein_embedding = self.protein_embedding(protein_index)
        combined = torch.cat((ligand_embedding, protein_embedding), dim=1)
        predictions = self.fc(combined)
        
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions.squeeze(), labels)
                
        return {"loss": loss, "logits": predictions} if loss is not None else predictions

def generate_ligand_embedding(smiles, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model(**tokenized)
        embedding = output.logits.mean(dim=1)
    return embedding.squeeze().cpu()

def generate_affinity_dataset(data, tokenizer, model):
    protein_sequences = [
        "JAK3_HUMAN",
        "JAK2_HUMAN",
        "JAK1_HUMAN",
        "TYK2_HUMAN",
        "CCR5_HUMAN",
        "TEC_HUMAN",
        "CCR5_MOUSE",
        "PDE4A_HUMAN",
        "MERTK_HUMAN",
    ]
    data = data[data['Target Name'].isin(protein_sequences)]
    data.fillna(0, inplace=True)

    encoder = OrdinalEncoder()
    ic50_values = torch.tensor([
        float(str(ic).replace("<", "").replace(">", "")) for ic in data['IC50 (nM)']
    ], dtype=torch.float32)

    encoder.fit(data[['Target Name']])
    data['Target Index'] = encoder.transform(data[['Target Name']]).astype(int)
    
    ligand_embeddings = []
    for smiles in tqdm(data['ligand']):
        embedding = generate_ligand_embedding(smiles, tokenizer, model)
        ligand_embeddings.append(embedding)
    ligand_embeddings = torch.stack(ligand_embeddings)

    protein_indices = torch.tensor(data['Target Index'].values, dtype=torch.int64)

    dataset_ic = Dataset.from_dict({
        "ligand_embedding": ligand_embeddings.numpy(),
        "protein_index": protein_indices.numpy(),
        "labels": ic50_values.numpy()
    })

    # Save the dataset
    dataset_ic.save_to_disk("data/ic50_dataset")
    return dataset_ic

def train_affinity(dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        logging_dir="./logs",
        fp16=False,
        bf16=False,
        weight_decay=0.1, 
        learning_rate=5e-4,
        logging_steps=1,
        save_total_limit=1,
        remove_unused_columns=False,
        label_names=["labels"]
    )

    train_size = int(0.8 * len(dataset))
    data_dict = dataset.train_test_split(train_size=train_size)
    train_dataset = data_dict['train']
    eval_dataset = data_dict['test']

    num_proteins = len(set(dataset["protein_index"]))
    embedding_size = len(dataset["ligand_embedding"][0])
    model = HFIC50Predictor(embedding_size, num_proteins)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IC50DataCollator(),
    )

    trainer.train()
    return model
