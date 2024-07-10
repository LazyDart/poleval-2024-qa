import pytest
import pandas as pd

import os

from .. import poquad


def identical_df_test(df1, df2):
    return ((df1 == df2).all(axis=None) 
        and (df1.columns == df2.columns).all() 
        and (df1.index == df2.index).all() 
        and (df1.index.name == df2.index.name)),


values = [
    [
    'Jan Paweł II',
    'W odpowiedzi na szeroko rozpowszechniony kult Jana Pawła II, po jego śmierci rozpowszechnił się fenomen internetowy polegający na tworzeniu memów nazywanych cenzopapami; cechują je: obrażanie papieża; wyśmiewanie jego przesadnego uwielbienia.',
    'Co było powodem fali popularności cenzopap?',
    False,
    {'answer_start': [16],
    'text': ['szeroko rozpowszechniony kult Jana Pawła II']}
    ],
    [
    'Jan Paweł II',
    'W odpowiedzi na szeroko rozpowszechniony kult Jana Pawła II, po jego śmierci rozpowszechnił się fenomen internetowy polegający na tworzeniu memów nazywanych cenzopapami; cechują je: obrażanie papieża; wyśmiewanie jego przesadnego uwielbienia.',
    'Co doprowadziło do kultu Jana Pawła II?',
    True,
    {'answer_start': [], 'text': []}
    ]
]

ideal_df = pd.DataFrame(values, columns=["title", "context", "question", "is_impossible", "answers"], index=pd.Series(["1", "2"], name="id"))

def test_read_poquad_manually_downloaded():
    df = poquad.read_poquad_manually_downloaded(f"{os.path.dirname(__file__)}/poquad_test_file.json")

    assert identical_df_test(df, ideal_df), "DataFrames are not the same after loading data with read_poquad_manually_downloaded function."


values = [
    ['kontekst: Jan Paweł II  W odpowiedzi na szeroko rozpowszechniony kult Jana Pawła II, po jego śmierci rozpowszechnił się fenomen internetowy polegający na tworzeniu memów nazywanych cenzopapami; cechują je: obrażanie papieża; wyśmiewanie jego przesadnego uwielbienia.  pytanie: Co było powodem fali popularności cenzopap?',
    'odpowiedź: szeroko rozpowszechniony kult Jana Pawła II'],
    ['kontekst: Jan Paweł II  W odpowiedzi na szeroko rozpowszechniony kult Jana Pawła II, po jego śmierci rozpowszechnił się fenomen internetowy polegający na tworzeniu memów nazywanych cenzopapami; cechują je: obrażanie papieża; wyśmiewanie jego przesadnego uwielbienia.  pytanie: Co doprowadziło do kultu Jana Pawła II?',
    'odpowiedź: [BRAK_ODPOWIEDZI]']
]

ideal_input_df = pd.DataFrame(values, columns=["input_text", "target_text"])

def test_dataset_into_str_input():
    df_from_ideal = poquad.dataset_into_str_input(ideal_df)
    df_from_test_file = poquad.dataset_into_str_input(
        poquad.read_poquad_manually_downloaded(f"{os.path.dirname(__file__)}/poquad_test_file.json")
    )

    assert identical_df_test(df_from_ideal, ideal_input_df), "Error encountered while processing data into input and target strings."
    assert identical_df_test(df_from_test_file, ideal_input_df), "Error encountered between loading manual poquad data and processing it into input and target strings."