def read_input():
    with open("input.txt", "r", encoding="utf-8") as f:
        return f.read()


def encode(char_to_index, string):
    return [char_to_index[c] for c in string]


def decode(index_to_char, indices):
    return "".join([index_to_char[i] for i in indices])


def train():
    text = read_input()
    vocabulary = sorted(set(text))
    print(vocabulary)
    print(len(vocabulary))
    char_to_index = {c: i for i, c in enumerate(vocabulary)}
    print(encode(char_to_index, "hii there"))
    print(decode(vocabulary, encode(char_to_index, "hii there")))


if __name__ == "__main__":
    train()
