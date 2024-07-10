import pandas as pd

from Levenshtein import distance

import re


def bin_f1_score(df):
    tp = (df["is_impossible"] & df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()
    fp = ((~df["is_impossible"]) & df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()
    fn = (df["is_impossible"] & ~df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def rm_answer_prefix(text):
    return re.sub(r"odpowiedź: ", "", text)


def normalized_levenshtein(text1, text2, *args, **kwargs):
    if len(text1) + len(text2) != 0:
        return distance(text1, text2, *args, **kwargs)/(len(text1) + len(text2))
    else:
        return 0