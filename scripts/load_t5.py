"""
Script loading PLT5 Model from Huggingface to ../models folder if not already present
"""

import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from argparse import ArgumentParser


def download_plt5(base=True, colab=False):

    root = "../" if not colab else "./poleval-2024-qa/"

    model_folder = os.path.join(root, 'models')

    # Create directory if not present
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    kind = "base" if base else "large"

    # Load T5 Model
    tokenizer = T5Tokenizer.from_pretrained(f'allegro/plt5-{kind}')
    model = T5ForConditionalGeneration.from_pretrained(f'allegro/plt5-{kind}')

    # Save Model
    tokenizer.save_pretrained(model_folder)
    model.save_pretrained(model_folder)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base", action="store_true", help="Download base model")
    parser.add_argument("--colab", action="store_true", help="Running on Colab")

    args = parser.parse_args()

    download_plt5(args.base, args.colab)