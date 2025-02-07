# Multi-Label Multi-Class Text-Span Classification with PyTorch 

This project implements a PyTorch-based solution for a multi-label, multi-class text-span classification task, specifically designed for named entity role assignment in news articles. The model handles both English and Portuguese text.

## Problem Definition

Given a news article and a list of named entity (NE) mentions within the article, the task is to assign one or more roles to each mention from a predefined taxonomy.  The roles are hierarchical, consisting of main roles (Protagonist, Antagonist, Innocent) and fine-grained roles (e.g., Guardian, Terrorist, Victim). This is a *multi-label* problem because a mention can have multiple fine-grained roles, and *multi-class* because each mention must be classified into one of the main/fine-grained role combinations.

## Data Format

*   **Labels:** `./dataset/training_data/<Language_code>/subtask-1-annotations.txt`
    *   Format: `article_id entity_mention start_offset end_offset main_role fine-grained_roles`
    *   `article_id`:  File name of the input article (e.g., `EN_UA_103861.txt`).
    *   `entity_mention`: The text of the entity mention (e.g., "China").
    *   `start_offset`, `end_offset`: Character offsets indicating the mention's location in the article.
    *   `main_role`:  One of "Protagonist", "Antagonist", or "Innocent".
    *   `fine-grained_roles`: A tab-separated list of fine-grained roles (e.g., "Instigator", "Terrorist\tDeceiver").

*   **Articles:** `./dataset/training_data/<Language_code>/raw-documents`
    *   Plain text files containing the news articles.

## Taxonomy of Roles

*   **Protagonist:** Guardian, Martyr, Peacemaker, Rebel, Underdog, Virtuous
*   **Antagonist:** Instigator, Conspirator, Tyrant, Foreign Adversary, Traitor, Spy, Saboteur, Corrupt, Incompetent, Terrorist, Deceiver, Bigot
*   **Innocent:** Forgotten, Exploited, Victim, Scapegoat

## Architecture

The solution is built around the `RoleClassifier` class, a PyTorch `nn.Module` designed for multi-task learning. It simultaneously performs two classification tasks:

1.  **Main Role Classification:** Single-label, multi-class classification (one of Protagonist, Antagonist, or Innocent).
2.  **Fine-Grained Role Classification:** Multi-label classification (zero or more of the 22 fine-grained roles).

Here's a detailed breakdown of the `RoleClassifier`'s structure:

### `RoleClassifier`

1.  **Initialization (`__init__`)**:

    *   **Pre-trained Transformer (`self.bert`):** The foundation of the model is a pre-trained transformer (default: `xlm-roberta-base`, a multilingual XLM-RoBERTa model).  It can be easily switched to other multilingual transformers like `bert-base-multilingual-cased`. This component provides strong contextualized word embeddings.
    *   **Hidden Size:**  The dimensionality of the transformer's hidden states (typically 768 for base models) is retrieved (`self.bert.config.hidden_size`).
    *   **Dropout (`self.dropout`):**  A dropout layer is included for regularization (default dropout probability: 0.3). This helps prevent the model from overfitting the training data.
    *   **Main Role Classifier (`self.main_classifier`):**  A linear layer (`nn.Linear`) that takes the transformer's output (hidden size) and projects it down to the number of main roles (3). This is the classification head for the *main role*.
    *   **Fine-Grained Role Classifier (`self.fine_classifier`):** Another linear layer (`nn.Linear`) that takes the transformer's output and projects it down to the number of fine-grained roles (22). This is the classification head for the *fine-grained roles*.  This is specifically designed for *multi-label* classification.

2.  **Forward Pass (`forward`)**:

    *   **Input:** The `forward` method takes two inputs:
        *   `input_ids`: Token IDs from the tokenizer.
        *   `attention_mask`: A mask indicating which tokens are padding.
    *   **Transformer Output:**  The input is passed through the pre-trained transformer (`self.bert`). This generates hidden states for each token.
    *   **`[CLS]` Token Representation:** The representation of the `[CLS]` token (the first token, used for classification) is extracted from the *last* hidden layer (`outputs.last_hidden_state[:, 0, :]`).
    *   **Dropout Application:** Dropout is applied to the `[CLS]` token's representation.
    *   **Classification Heads:**
        *   The `[CLS]` representation is passed through the `main_classifier` to produce logits for the *main role* classification.
        *   The `[CLS]` representation is also passed through the `fine_classifier` to produce logits for the *fine-grained role* classification.
    *   **Output:** The method returns a tuple containing the logits for both the main and fine-grained role classifications.

**Key Architectural Design Choices:**

*   **Transformer Foundation:** The use of a pre-trained, multilingual transformer (XLM-RoBERTa or Multilingual BERT) is crucial.  These models are trained on vast amounts of text and capture rich contextual information, essential for understanding subtle role distinctions in different languages.
*   **Dual Classification Heads:** Separate linear layers for main and fine-grained roles allow the model to learn task-specific features. This separation is vital because the tasks have different output spaces (single-label vs. multi-label).
*   **Multi-Task Learning:**  The model is trained to predict both role types simultaneously. This joint training can improve performance as the tasks are related.  The overall loss is a weighted sum of the main role loss and the fine-grained role loss.
* **Special Tokens:** Special tokens `<entity>` and `</entity>` are added during the Data Preprocessing in RoleDataset Class to improve the focus on the entities.
* **Token Embedding Resizing**: After adding the tokens, token embedding is resized.
* **Loss Function Adaptation**: `CrossEntropyLoss` for the main role classification and `BCEWithLogitsLoss` for the multi-label, fine grained role classification.
* **Optimizer & Scheduler**: The AdamW optimizer, combined with a linear learning rate scheduler with warmup, provides robust and efficient training.
* **Early Stopping**: Implemented to prevent overfitting and save training time based on validation loss monitoring.
* **Comprehensive Metrics**: Tracks and reports a range of metrics (accuracy, precision, recall, F1-score - macro/micro) suitable for both single-label and multi-label classification. The 'exact match ratio' provides a strict measure of performance for multi-label predictions.
