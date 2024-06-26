"""
Script loading PLT5 Model from Huggingface to ../models folder if not already present
"""

import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from argparse import ArgumentParser


def download_plt5(large=False, colab=False):

    kind = "large" if large else "base"
    root = "../" if not colab else "./poleval-2024-qa/"

    model_folder = os.path.join(root, f'models/plt5-original-{kind}')

    # Create directory if not present
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Load T5 Model
    tokenizer = T5Tokenizer.from_pretrained(f'allegro/plt5-{kind}')
    model = T5ForConditionalGeneration.from_pretrained(f'allegro/plt5-{kind}')

    # Save Model
    tokenizer.save_pretrained(model_folder)
    model.save_pretrained(model_folder)


if __name__ == "__main__":

    # Running it from command line
    # On Local Machine
    # cd to scripts folder
    # python load_t5.py

    # On Colab
    # !python ./poleval-2024-qa/scripts/load_t5.py --colab

    parser = ArgumentParser()
    parser.add_argument("--large", action="store_true", help="Download base model")
    parser.add_argument("--colab", action="store_true", help="Running on Colab")

    args = parser.parse_args()

    download_plt5(args.base, args.colab)