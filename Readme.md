
# PolEval 2024 - Polish Question-Answering System

## Project Overview

This repository is dedicated to the development of a system for open-domain machine reading comprehension for question answering in Polish, as part of the PolEval 2024 competition. The system will process a question and a paired passage to generate an answer or determine if the question is unanswerable.

## Directory Structure

```
.
├── data
├── models
├── notebooks
├── outputs
├── Readme.md
├── scripts
└── secrets
    └── secret_repo_token
```

- **data:** Contains the dataset for training and evaluation.
- **models:** Directory for saving trained models.
- **notebooks:** Jupyter notebooks for data exploration and model training, including work done in Google Colab.
- **outputs:** Generated outputs from the models, including prediction files.
- **scripts:** Contains scripts for data preprocessing, training, and evaluation.
- **secrets:** Stores sensitive information like repository tokens (ensure this is secure).

## Usage

The project will utilize both local hardware and Google Colab for training tasks. Initial data preprocessing and smaller tasks will be handled locally, while model training will be conducted on Google Colab, leveraging its computational resources.

## Development Environment

Work will be managed across local files and Google Colab environments. Notebooks in the `notebooks` directory will reflect this hybrid development approach, showing the steps and experiments conducted in Colab.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details (to be added).

## Future Updates

This README will be updated regularly as the project progresses and more details become available.
