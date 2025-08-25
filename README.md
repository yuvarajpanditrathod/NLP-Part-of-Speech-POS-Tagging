# NLP Part-of-Speech (POS) Tagging with Adversarial Training

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Processing-green.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![POS Tagging](https://img.shields.io/badge/POS-Tagging-yellow.svg)](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
[![Adversarial Training](https://img.shields.io/badge/Adversarial-Training-critical.svg)](https://arxiv.org/abs/1412.6572)
[![Dataset](https://img.shields.io/badge/Dataset-CSV-lightgrey.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


This repository contains the implementation of advanced Part-of-Speech (POS) tagging models using BiLSTM-CRF and BERT-CRF architectures enhanced with Virtual Adversarial Training (VAT). The project explores state-of-the-art approaches to sequence labeling in Natural Language Processing.

## 🚀 Run on Google Colab

🔗 [Open the Notebook in Google Colab](https://colab.research.google.com/drive/1xD_twIV6z6kD_YH3LAM0j-4j7Q6I7mUf)


## 🎯 Project Overview

Part-of-Speech tagging is a fundamental task in Natural Language Processing that involves assigning grammatical categories (e.g., noun, verb, adjective) to each word in a sentence. This project implements and compares multiple deep learning approaches:

1. **BiLSTM-CRF with Virtual Adversarial Training (VAT)**
2. **BERT-CRF with Virtual Adversarial Training**
3. **Traditional BiLSTM-CRF baseline**

### Key Features 

- 🚀 **State-of-the-art Models**: Implementation of BiLSTM-CRF and BERT-CRF architectures
- 🔄 **Virtual Adversarial Training**: Enhanced robustness through adversarial perturbations
- 📊 **Comprehensive Evaluation**: Detailed performance metrics and confusion matrix analysis
- 📚 **Multiple Datasets**: Support for Universal Dependencies (UD) CoNLL-U format
- 🎨 **Visualization**: Performance visualization and error analysis tools


## 🛠️ Installation

### Prerequisites

- Python 3.10+ or higher
- CUDA-compatible GPU (recommended for faster training)

### Required Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pytorch-crf
pip install conllu
pip install seqeval
pip install scikit-learn
pip install matplotlib
pip install numpy pandas
```

### Alternative Installation

All dependencies can be installed by running the first cell of the Jupyter notebook:

```python
!pip install conllu seqeval pytorch-crf scikit-learn matplotlib transformers
```

## 🚀 Quick Start

### 1. Data Preparation

The project uses Universal Dependencies CoNLL-U format files. You need to download the English EWT dataset:

```bash
# Download UD English-EWT dataset
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu
```

### 2. Running the Models

Open the `NLP-Colab-Notebook.ipynb` in Jupyter Notebook or Google Colab and execute the cells sequentially:

```python
# Load and preprocess data
train_sentences, train_tags = read_conllu("en_ewt-ud-train.conllu")
dev_sentences, dev_tags = read_conllu("en_ewt-ud-dev.conllu")
test_sentences, test_tags = read_conllu("en_ewt-ud-test.conllu")

# Build vocabulary
word2idx, tag2idx, idx2tag = build_vocab(train_sentences, train_tags)

# Train BiLSTM-CRF with VAT
model = BiLSTM_CRF(vocab_size, tagset_size, pad_idx=pad_idx)
train_model_vat(model, X_train, y_train, optimizer, tag_pad_idx=tag2idx["<PAD>"], epochs=5)
```

### 3. BERT-CRF with VAT

```python
# Initialize BERT-CRF model
model = BERT_CRF_VAT(len(tag2idx))
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Train with Virtual Adversarial Training
# Training loop with combined CRF loss and VAT loss
```

## 🏗️ Model Architectures

### 1. BiLSTM-CRF with VAT

```
Input → Embedding → BiLSTM → Linear → CRF → Output
         ↓
    VAT Perturbation
```

**Key Components:**
- **Embedding Layer**: Word embeddings with padding support
- **BiLSTM**: Bidirectional LSTM for context modeling
- **CRF Layer**: Conditional Random Field for sequence labeling
- **VAT**: Virtual Adversarial Training for regularization

### 2. BERT-CRF with VAT

```
Input → BERT Tokenizer → BERT Encoder → Linear → CRF → Output
                         ↓
                   VAT Perturbation
```

**Key Components:**
- **BERT Encoder**: Pre-trained BERT-base-cased
- **Token Alignment**: Subword to word alignment
- **CRF Layer**: Structured prediction
- **VAT**: Adversarial perturbations on embeddings

## 📊 Performance Metrics

The models are evaluated using:

- **Token-level Accuracy**: Individual token classification accuracy
- **Sequence-level Accuracy**: Complete sentence accuracy
- **F1-Score**: Macro and micro-averaged F1 scores
- **Confusion Matrix**: Detailed error analysis per POS tag
- **Classification Report**: Precision, recall, and F1 per tag

### Example Results

```
              precision    recall  f1-score   support

       NOUN       0.92      0.94      0.93      1234
       VERB       0.89      0.87      0.88       987
        ADJ       0.85      0.83      0.84       654
        ...       ...       ...       ...       ...

avg / total       0.91      0.91      0.91     8765
```

## 🔬 Virtual Adversarial Training (VAT)

Virtual Adversarial Training enhances model robustness by:

1. **Generating Adversarial Examples**: Creating small perturbations in embedding space
2. **Consistency Regularization**: Ensuring consistent predictions for perturbed inputs
3. **Improved Generalization**: Better performance on unseen data

### VAT Implementation

```python
def compute_vat_loss(model, x, mask, xi=1e-6, epsilon=3.5, num_power_iter=1):
    # Generate adversarial perturbation
    d = torch.randn_like(embeds)
    for _ in range(num_power_iter):
        d = xi * F.normalize(d, p=2, dim=-1)
        # ... (see notebook for full implementation)
    return vat_loss
```

## 📈 Training Process

### Hyperparameters

| Parameter | BiLSTM-CRF | BERT-CRF |
|-----------|------------|----------|
| Learning Rate | 0.001 | 2e-5 |
| Batch Size | 1 | 4 |
| Hidden Dim | 256 | 768 |
| Embedding Dim | 100 | 768 |
| VAT Alpha | 0.5 | 0.5 |
| VAT Epsilon | 3.5 | 2.0 |

### Training Loop

```python
for epoch in range(epochs):
    for batch in dataloader:
        # Standard loss (CRF)
        crf_loss = model(input_ids, attention_mask, labels)
        
        # VAT loss
        vat_loss = compute_vat_loss(model, input_ids, attention_mask)
        
        # Combined loss
        total_loss = crf_loss + alpha * vat_loss
        
        total_loss.backward()
        optimizer.step()
```

## 📋 Dataset Information

### Universal Dependencies Format

The project uses CoNLL-U format files with the following structure:

```
1    They    they    PRON    PRP    Case=Nom|Number=Plur    2    nsubj    _    _
2    buy     buy     VERB    VBP    Number=Plur|Person=3|Tense=Pres    0    root    _    _
3    books   book    NOUN    NNS    Number=Plur    2    obj    _    _
```

### Supported POS Tags

The Universal POS tagset includes:
- **Open class words**: NOUN, VERB, ADJ, ADV
- **Closed class words**: PRON, DET, ADP, NUM, CONJ, PRT
- **Other**: PUNCT, X (unknown), SYM (symbols)

## 🔧 Customization

### Adding New Models

To add a new model architecture:

1. Create a new model class inheriting from `nn.Module`
2. Implement forward pass with CRF compatibility
3. Add VAT support if desired
4. Update training loop accordingly

### Custom Datasets

To use custom datasets:

1. Convert data to CoNLL-U format
2. Update file paths in data loading functions
3. Adjust vocabulary building if needed

## 📚 Research Paper

This implementation is based on research documented in `NLP-Conference-Paper.pdf`. The paper covers:

- **Theoretical Background**: CRF and VAT foundations
- **Experimental Setup**: Dataset preparation and model configuration
- **Results Analysis**: Comparative study of different approaches
- **Future Work**: Potential improvements and extensions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, suggestions, or collaborations:

- **Email**: [yuvarajpanditrathod@gmail.com](mailto:yuvarajpanditrathod@gmail.com)  
- **GitHub**: [yuvarajpanditrathod](https://github.com/yuvarajpanditrathod)  
- **LinkedIn**: [Yuvaraj P Rathod](https://linkedin.com/in/yuvarajpanditrathod)  


## 🙏 Acknowledgments

- **Universal Dependencies**: For providing high-quality linguistic datasets
- **Hugging Face**: For the Transformers library and pre-trained models
- **PyTorch Team**: For the excellent deep learning framework
- **Research Community**: For foundational work in POS tagging and adversarial training
 find this project helpful, please consider giving it a star!**
