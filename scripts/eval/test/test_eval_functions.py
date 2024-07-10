import pytest
import pandas as pd

import os

from .. import eval



# def bin_f1_score(df):
#     tp = (df["is_impossible"] & df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()
#     fp = ((~df["is_impossible"]) & df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()
#     fn = (df["is_impossible"] & ~df["gen_text"].str.contains(r"\[BRAK\_ODPOWIEDZI\]")).sum()

#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#     return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# def rm_answer_prefix(text):
#     return re.sub(r"odpowiedź: ", "", text)


# def normalized_levenshtein(text1, text2, *args, **kwargs):
#     return distance(text1, text2, *args, **kwargs)/(len(text1) + len(text2))




# Write pytest tests to functions above

def test_bin_f1_score():
    df = pd.DataFrame([[True, "odpowiedź: [BRAK_ODPOWIEDZI]"], [False, "odpowiedź: [BRAK_ODPOWIEDZI]"]], columns=["is_impossible", "gen_text"])
    assert round(eval.bin_f1_score(df), 3) == 0.667, "Invalid calculation of f1 score."

    df = pd.DataFrame([[True, "odpowiedź: [BRAK_ODPOWIEDZI]"], [True, "odpowiedź: [BRAK_ODPOWIEDZI]"]], columns=["is_impossible", "gen_text"])
    assert round(eval.bin_f1_score(df), 3) == 1.0, "Invalid calculation of f1 score."

    df = pd.DataFrame([[False, "odpowiedź: [BRAK_ODPOWIEDZI]"], [False, "odpowiedź: [BRAK_ODPOWIEDZI]"]], columns=["is_impossible", "gen_text"])
    assert round(eval.bin_f1_score(df), 3) == 0.0, "Invalid calculation of f1 score."


def test_rm_answer_prefix():
    assert eval.rm_answer_prefix("odpowiedź: [BRAK_ODPOWIEDZI]") == "[BRAK_ODPOWIEDZI]", "Error in rm_answer_prefix function."
    assert eval.rm_answer_prefix("odpowiedź: test") == "test", "Error in rm_answer_prefix function."
    assert eval.rm_answer_prefix("odpowiedź: ") == "", "Error in rm_answer_prefix function."
    assert eval.rm_answer_prefix("") == "", "Error in rm_answer_prefix function."


def test_normalized_levenshtein():
    assert round(eval.normalized_levenshtein("test", "test"), 3) == 0.0, "Error in normalized_levenshtein function."
    assert round(eval.normalized_levenshtein("test", "test2"), 3) == 0.111, "Error in normalized_levenshtein function."
    assert round(eval.normalized_levenshtein("test", ""), 3) == 1, "Error in normalized_levenshtein function."
    assert round(eval.normalized_levenshtein("", "test"), 3) == 1, "Error in normalized_levenshtein function."
    assert round(eval.normalized_levenshtein("", ""), 3) == 0.0, "Error in normalized_levenshtein function."