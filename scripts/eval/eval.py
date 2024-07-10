import pandas as pd
import re

def bin_f1_score(df):
    tp = (df["is_impossible"] & df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()
    fp = ((~df["is_impossible"]) & df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()
    fn = (df["is_impossible"] & ~df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * (precision * recall) / (precision + recall)

def rm_answer_prefix(text):

    return re.sub(r"odpowiedź: ", "", text)