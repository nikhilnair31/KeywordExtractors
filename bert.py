# from transformers import pipeline

# nlp = pipeline("sentiment-analysis")
# print(nlp("I hate you"))
# print(nlp("I love you"))

# nlp = pipeline("ner")
# sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \ "close to the Manhattan Bridge which is visible from the window."
# print(nlp(sequence))

from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

model = TFAutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

label_list = ["O","B-MISC", "I-MISC","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC"]

sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge."

tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="tf")

outputs = model(inputs)[0]
predictions = tf.argmax(outputs, axis=2)

print(tokens)
print(inputs)
print(outputs)
print(predictions)
print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])