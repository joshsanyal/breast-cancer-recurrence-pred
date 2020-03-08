import torch
import transformers
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


model = BertForSequenceClassification.from_pretrained()
