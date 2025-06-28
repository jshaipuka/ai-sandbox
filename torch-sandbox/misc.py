import torch


def exp():
    matrix = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
    ])

    vector = torch.tensor([
        1,
        2
    ])

    print(matrix)
    print(matrix.transpose(0, 1))

    # print(matrix.shape)
    # print(vector.shape)
    #
    # print(matrix)
    # print(vector)
    #
    # print(matrix + vector.T)


# ----

# https://docs.pytorch.org/docs/stable/generated/torch.linalg.eig.html
# Ctrl+J or F1 (https://stackoverflow.com/q/11053144/1862286)
def exp2():
    a = torch.tensor([
        [1.0, 2],
        [3, 4]
    ])
    l, v = torch.linalg.eig(a)
    print(l)
    print(v)
    print(v @ torch.diag(l) @ torch.inverse(v))


def exp3():
    # a is a singular matrix, because its det is 0.
    a = torch.tensor([
        [2.0, 1],
        [8, 4]
    ])
    l, v = torch.linalg.eig(a)
    # The matrix is singular if and only if any of the eigenvalues are zero.
    print(l)
    print(v)
    print(v @ torch.diag(l) @ torch.inverse(v))
    print(torch.det(a))
    # print(torch.linalg.inv(a))
    print(torch.linalg.pinv(a))


def exp4():
    a = torch.diag(torch.tensor([1.0, 2, 4]))
    print(torch.inverse(a))
    print(torch.linalg.pinv(a))


def exp5():
    a = torch.cat([torch.diag(torch.tensor([1.0, 2, 4])), torch.zeros(3, 2)], dim=1)
    # You can't invert a rectangular matrix, but you can compute the pseudoinverse (Moore-Penrose inverse) of it
    # print(torch.inverse(a))
    print(a)
    print(torch.linalg.pinv(a))


exp3()
