# create dataset class
from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time


class Dataset4Summarization(Dataset):
	def __init__(self, data, tokenizer, max_length=1024*3, chunk_length =1024):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.chunk_length = chunk_length

	def __len__(self):
		return len(self.data)
	
	def chunking(self, text):
		chunks = []
		for i in range(0, self.max_length, self.chunk_length):
			chunks.append(text[i:i+self.chunk_length])
		return chunks

	def __getitem__(self, idx):
		sample = self.data[idx]
		inputs = self.tokenizer(sample, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

		list_chunk = self.chunking(inputs['input_ids'].squeeze())
		list_attention_mask = self.chunking(inputs['attention_mask'].squeeze())


		return {
			'list_input_ids': list_chunk,
			'list_att_mask' : list_attention_mask,
		}
	

def process_data_infer(data):
	single_documents = data.get('single_documents', [])

	
	result = []
	for doc in single_documents:
		raw_text = doc.get('raw_text', '')
		result.append(raw_text)

	return " ".join(result)


def processing_data_infer(input_file):
	all_results = []
	
	with open(input_file, 'r', encoding='utf-8') as file:
		for line in file:
			data = json.loads(line.strip())
			result = process_data_infer(data)
			all_results.append(result)

	return all_results

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load("./weight_cp19_model.pth", map_location=torch.device('cpu')))

# For other demo purpose, you just need to make sure data is list of documents [document1, document2]

# batch_size need to be 1,
@torch.no_grad()
def infer_2_hier(model, data_loader, device, tokenizer):
    model.eval()
    start = time.time()
    all_summaries = []
    for iter in data_loader:
        summaries = []
        inputs = iter['list_input_ids']
        att_mask = iter['list_att_mask']
        
        for i in range(len(inputs)):
            # Check if the input tensor is all zeros
            if torch.all(inputs[i] == 0):
                # If the input is all zeros, skip this iteration
                continue
            else:
                summary = model.generate(inputs[i].to(device),
                                         attention_mask=att_mask[i].to(device),
                                         max_length=128,
                                         num_beams=12,
                                         num_return_sequences=1)
                summaries.append(summary)
        summaries = torch.cat(summaries, dim = 1)
        for k in summaries:
                all_summaries.append(tokenizer.decode(k, skip_special_tokens=True))

    
    end = time.time()
    print(f"Time: {end-start}")
    return all_summaries

def vit5_infer(data):
	print(data)
	dataset = Dataset4Summarization(data, tokenizer)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
	result = infer_2_hier(model, data_loader, device, tokenizer)
	return result