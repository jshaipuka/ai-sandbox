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
    # batch = torch.tensor([
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3],
    #     [1, 2, 3]
    # ], dtype=torch.float32)
    # model = nn.Sequential(
    #     nn.Linear(3, 2),
    #     nn.ReLU()
    # )
    # print(batch.shape)
    # print(model)
    # print(model(batch))
    # export_model(model, batch)


if __name__ == '__main__':
    main()
