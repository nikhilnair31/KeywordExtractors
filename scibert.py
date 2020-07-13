import transformers

tokenizer = AutoTokenizer.from_pretrained('Data/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('Data/scibert_scivocab_uncased')

text = "Hello, y'all! How are you 😁 ?"
output = tokenizer.encode(text)
print(output)