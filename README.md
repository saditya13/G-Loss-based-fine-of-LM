# G-Loss: Graph-Driven-Fine-tuning-LM
This repo contains code for [link to paper]

## Introduction

Fine-tuning pre-trained language models like *SBERT* often relies on instance- or pair-wise loss functions, which fail to capture global semantic relationships and incur high computational costs.

We propose **G-Loss**, a graph-driven loss function computed using label propagation, capturing holistic semantic relationships leading to better embedding alignment. 

We evaluate **G-Loss** on five datasets: MR for sentiment classification, R8 and R52 for topic classification, *Ohsumed* for medical document classification, and 20NG for news categorization.

**G-Loss** enabled *SBERT* to outclass or closely match the performance of *TextGCN*, *BertGCN-BERT*, and *BertGCN-RoBERTa*.

<br>

To compute *G-Loss*:
* Embedding Extraction: Embed the input data using the specified language model.
* Similarity Graph Construction: Construct a graph based on the similarity of embeddings within a mini-batch.
* Label Masking: Mask a subset of node labels for _Label Propagation_.
* Iterative Label Propagation (LPA): Perform iterative label propagation to infer labels for the masked nodes.
* Loss Computation: Calculate the cross-entropy loss between the LPA-predicted labels and the ground truth labels of the masked nodes.

The computed loss is propagated to the language model, which then updates itself.
A key feature of this approach is that the graph evolves dynamically over fine-tuning epochs, improving semantic representation as embeddings become more refined.

## Main results of paper

## Dependencies

### Prerequisites

*   Python 3.9+ (Recommended)

### Setting up the Environment

1.  **Create a virtual environment:**

    *   **Using conda (Recommended):**

        ```bash
        conda create -n my_env python=3.10
        conda activate my_env
        ```

    *   **Using venv (Built-in to Python):**

        ```bash
        python3 -m venv my_env  # Or python -m venv my_env depending on your setup
        source my_env/bin/activate  # On Windows: my_env\Scripts\activate
        ```

    *   **Using virtualenv (If you prefer this):**

        ```bash
        pip install virtualenv
        virtualenv my_env
        source my_env/bin/activate  # On Windows: my_env\Scripts\activate
        ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```


### Library Packages

This project requires the following Python packages:

*   `torch == 2.5.1`
*   `optuna == 4.0.0`
*   `pandas == 2.2.3`
*   `scikit-learn == 1.3.2`
*   `transformers == 4.48.2`
*   `sentence-transformers == 3.3.1`

You can install these packages using pip:

```bash
pip install torch>=2.5.1 optuna>=4.0.0 pandas>=2.2.3 scikit-learn>=1.3.2 matplotlib>=3.10.0 transformers>=4.48.2 sentence-transformers>=3.3.1

```

## Usage
1. step1
2. step 2
3. step 3

## Acknowledgement

## Citation

