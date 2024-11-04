import numpy as np


def f(words, current_text):
    pre_pro_words = []
    pre_pro_words_distinct = []
    try:
        for i, w in enumerate(words):
            if w[0] == " ":
                pre_pro_words.append(i - 1)
                if len(pre_pro_words) >= 2:
                    pre_pro_words_distinct.append(
                        words[pre_pro_words[-2] + 1 : pre_pro_words[-1] + 1]
                    )
                else:
                    pre_pro_words_distinct.append(words[: pre_pro_words[-1] + 1])

        print(pre_pro_words)
        print(pre_pro_words_distinct)

        if len(pre_pro_words) == 0:
            return [], []

        pre_pro_words.pop(0)
        pre_pro_words_distinct.pop(0)
        pre_pro_words.append(len(words) - 1)
    except IndexError as e:
        raise IndexError from e

    print(pre_pro_words)
    print(pre_pro_words_distinct)

    if len(pre_pro_words) >= 2:
        pre_pro_words_distinct.append(
            words[pre_pro_words[-2] + 1 : pre_pro_words[-1] + 1]
        )
    else:
        pre_pro_words_distinct.append(words[: pre_pro_words[-1] + 1])

    print(pre_pro_words)
    print(pre_pro_words_distinct)
    return pre_pro_words, pre_pro_words_distinct


def f2(words, current_text):
    npw = np.array(words)
    # print(npw.view("<U1")[:: len(npw[0])])
    non_space_words = [i for i, w in enumerate(npw) if w[0] != " "]
    a = np.array(non_space_words)
    pre_pro_words = np.delete(np.arange(len(npw)), a - 1)
    if len(pre_pro_words) == 0:
        return [], []
    pre_pro_words_distinct = [words[0 : pre_pro_words[0] + 1]] + [
        words[x + 1 : pre_pro_words[i + 1] + 1]
        for i, x in enumerate(pre_pro_words[:-1])
    ]
    print(pre_pro_words)
    print(pre_pro_words_distinct)
    return list(pre_pro_words), pre_pro_words_distinct


# ex 1
words = [" Hi", " there", ","]
current_text = "Hi there,"
w, wd = f(words, current_text)
w2, wd2 = f2(words, current_text)
assert w == w2
assert wd == wd2

# ex 2
words = [" I", "'", "m", " doing", " well", " today", "!"]
current_text = "I'm doing well today!"
w, wd = f(words, current_text)
w2, wd2 = f2(words, current_text)
assert w == w2
assert wd == wd2

# ex 3
words = [" Now", ","]
current_text = " Now,"
w, wd = f(words, current_text)
w2, wd2 = f2(words, current_text)
assert w == w2
assert wd == wd2

# ex 4
words = ["?"]
current_text = "?"
w2, wd2 = f2(words, current_text)
w, wd = f(words, current_text)
assert w == w2
assert wd == wd2
