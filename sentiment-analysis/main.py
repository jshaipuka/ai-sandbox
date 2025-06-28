import spacy
import torch
from datasets import load_dataset, Dataset


def export_model(model, tensor):
    torch.onnx.export(model, tensor, "model.onnx")


def main():
    nlp = spacy.load('en_core_web_sm')
    dataset = load_dataset('Sp1786/multiclass-sentiment-analysis-dataset')
    train_dataset: Dataset = dataset['train']
    # limit = 10
    # print(train_dataset['text'][:limit])
    # print(train_dataset['label'][:limit])
    # print(train_dataset['sentiment'][:limit])

    ignore = {'SPACE', 'PUNCT', 'PART'}
    text: str = ' '.join(train_dataset['text'])
    nlp.max_length = len(text)
    doc = nlp(text)
    words: set[str] = {token.text.lower() for token in doc if token.pos_ not in ignore}
    print(len(words))


if __name__ == '__main__':
    main()
