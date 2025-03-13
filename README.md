# Sequence-to-Sequence Model with Bahdanau Attention

This repository contains a **Sequence-to-Sequence** (Seq2Seq) model with **Bahdanau Attention** for English-to-French translation. The model is implemented using `PyTorch` and follows a modular structure for improved readability, maintainability, and scalability.

## Features

**PyTorch Implementation**: Fully built using `PyTorch` for efficient deep learning workflows.

**Data Version Control (DVC)**: Manages training and evaluation pipelines effectively.

**Marian Tokenizer**: Uses `Helsinki-NLP/opus-mt-en-fr` for robust and efficient text processing.

**Experiment Tracking & Model Management**: Integrated with `MLflow` and `DagsHub` for seamless tracking.

**Containerized Deployment**: Docker images stored in `Amazon Elastic Container Registry (ECR)`.

**Scalable Deployment**: Model deployed on `Amazon Elastic Kubernetes Service (EKS)` for production readiness.

**Automated CI/CD**: End-to-end deployment automation using AWS and GitHub Actions.

## Prerequisites

Ensure the following dependencies and services are installed and configured:

- Python 3.10
- AWS Account
- AWS CLI
- Docker Desktop (for local image testing)
- DagsHub Account (for experiment tracking)
- Git & GitHub (for version control)

## Dataset

**Source:** [Language Translation (English-French)](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)

**Description:**
The dataset consists of:

- **English words/sentences**
- **French words/sentences**

## Model Architecture

The **Sequence-to-Sequence** model consists of the following components:

### 1. Encoder

- A multi-layer **Long Short-Term Memory (LSTM)** network processes input sequences and encodes them into `hidden state` representations.
- An `embedding layer` converts tokenized words into dense vector representations.
- The final `hidden states` and `cell states` are passed to the **decoder** for further processing.

### 2. Bahdanau Attention

- The **attention mechanism** enables the decoder to focus on relevant parts of the `encoder output` while generating each word in the target language.
- Computes **attention scores** using a combination of the decoder's `hidden state` (query) and `encoder outputs` (keys).
- Scores are normalized using softmax to obtain attention weights, which are used to compute a weighted sum of the encoder outputs (context vector).

### 3. Decoder

- A multi-layer **LSTM** that takes the previous token and `context vector` as input at each time step.
- The `context vector` from the **attention mechanism** assists in generating accurate translations by attending to relevant encoder states.
- The output is passed through a **fully connected layer** to predict the next `token` in the sequence.

### 4. Seq2Seq Model

- The complete model integrates the **encoder** and **decoder** modules.
- During **training**, the decoder receives ground-truth tokens as input for teacher forcing.
- The model outputs a probability distribution over the target vocabulary at each time step.

#### Model Summary

```text
Model(
  (encoder): Encoder(
    (embedding): Embedding(59515, 128)
    (lstm): LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2)
  )
  (decoder): Decoder(
    (embedding): Embedding(59515, 128)
    (lstm): LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2)
    (attention): BahdanauAttention(
      (w_Q): Linear(in_features=128, out_features=128, bias=True)
      (w_K): Linear(in_features=128, out_features=128, bias=True)
      (w_V): Linear(in_features=128, out_features=1, bias=True)
    )
    (fc): Linear(in_features=256, out_features=59515, bias=True)
  )
)
```

## Installation

Clone the repository:

```sh
git clone https://github.com/aakash-dec7/SequenceToSequence.git
cd SequenceToSequence
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Initialize DVC Pipeline

```sh
dvc init
```

## DVC Pipeline Stages

1. **Data Ingestion** - Fetches and stores the raw dataset.
2. **Data Validation** - Ensures data quality and integrity before processing.
3. **Data Preprocessing** - Cleans, tokenizes, and prepares the dataset for transformation.
4. **Data Transformation** - Converts processed data into a format suitable for model training.
5. **Model Definition** - Defines the model architecture.
6. **Model Training** - Trains the model using the transformed data.
7. **Model Evaluation** - Assesses the trained model’s performance.

### Run Model Training and Evaluation

```sh
dvc repro
```

The trained model will be saved in:

```sh
artifacts/model/model.pth
```

## Deployment

### Create an ECR Repository

Ensure that the Amazon ECR repository exists with the appropriate name as specified in `setup.py`:

```python
setup(
    name="seq2seq",
    version="1.0.0",
    author="Aakash Singh",
    author_email="aakash.dec7@gmail.com",
    packages=find_packages(),
)
```

### Create an EKS Cluster

Execute the following command to create an Amazon EKS cluster:

```sh
eksctl create cluster --name <cluster-name> \
    --region <region> \
    --nodegroup-name <nodegroup-name> \
    --nodes <number-of-nodes> \
    --nodes-min <nodes-min> \
    --nodes-max <nodes-max> \
    --node-type <node-type> \
    --managed
```

### Push Code to GitHub

Before pushing the code, ensure that the necessary GitHub Actions secrets are added under **Settings > Secrets and Variables > Actions**:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REGISTRY_URI`

Push the code to GitHub:

```sh
git add .
git commit -m "Initial commit"
git push origin main
```

### CI/CD Automation

GitHub Actions will automate the CI/CD process, ensuring that the model is built, tested, and deployed to **Amazon EKS**.

## Accessing the Deployed Application

Once deployment is successful:

1. Navigate to **EC2 Instances** in the **AWS Console**.
2. Go to **Security Groups** and update inbound rules to allow traffic.

Retrieve the external IP of the deployed service:

```sh
kubectl get svc
```

Copy the `EXTERNAL-IP` and append `:5000` to access the application:

```text
http://<EXTERNAL-IP>:5000
```

The Sequence-to-Sequence translation application is now deployed and accessible online.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
