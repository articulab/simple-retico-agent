from functools import partial

A = 0


def a(a, b):
    return a, b


aa = partial(a, 1)

print(aa(2))
