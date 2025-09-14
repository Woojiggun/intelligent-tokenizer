# Attention Needs No Vocabulary: Byte-Level Learning for Universal Tokenization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ggunio%2Fintelligent--tokenizer--v6-orange)](https://huggingface.co/ggunio/intelligent-tokenizer-v6)

*Title offered as an homage to "Attention Is All You Need" (Vaswani et al., 2017)*

## 🚀 Overview

A revolutionary approach to tokenization that operates directly on raw UTF-8 bytes without any vocabulary files or language-specific rules. Our model learns language patterns purely from data, achieving vocabulary-free tokenization across 204 languages.

### Key Features
- 🌍 **Universal**: Works with any UTF-8 encoded text (204 languages tested)
- 📦 **No Vocabulary**: Zero vocabulary files, pure byte-level processing
- ⚡ **Lightweight**: Only 105M parameters (vs. MegaByte's 1.5B)
- 🔄 **Streaming**: Real-time processing with 256-byte chunks
- 🎯 **Pure Learning**: No hardcoded rules, learns everything from data

## 📊 Performance

| Language | Reconstruction Accuracy | Notes |
|----------|------------------------|-------|
| English | 95% | Excellent performance |
| Korean | 70% (97% when trained alone) | Affected by multilingual transition |
| Japanese | 81% | Good for mixed scripts |
| Chinese | 7% | Needs more training |
| Average (204 languages) | 47% | POC stage |

*Note: Trained for only 24 hours on RTX 4070. Korean achieved 97% accuracy during dedicated training (epochs 1-20).*

## 🏗️ Architecture

```
Input Text → UTF-8 Bytes → 256-byte Chunks → Encoder (5 layers) → Decoder (6 layers) → Reconstruction
```

### Core Innovation
- **Hierarchical Relationship Learning**: Cross-attention at each embedding layer
- **Dynamic Semantic Grouping**: Learns byte grouping based on context
- **Vocabulary-Free Paradigm**: Enables true zero-shot cross-lingual transfer

## 🔗 Resources

- 📄 **Paper**: [Read on Zenodo](https://zenodo.org/records/17116281?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIyNWZiYTQyLWNiNGEtNDBmNi1iNTczLWVkMDJlNDI1YTQ1OSIsImRhdGEiOnt9LCJyYW5kb20iOiI0OWJkZWMzMjJjZTc3OTIwMTk4NTJlNTY1YmNjOGU1ZiJ9.Z_hXEp160tWBD5Qe2laQv1vhS4Js2a0R5BMWYs2PTG5vJMrc8l-BmPAIMya9O_HiN85jYZp-WOMOHg_DTHrg2A) | [PDF](Intelligent%20Tokenizer.pdf)
- 🤗 **Model**: [Hugging Face - ggunio/intelligent-tokenizer-v6](https://huggingface.co/ggunio/intelligent-tokenizer-v6)
- 🎮 **Live Demo**: [Try on Hugging Face Spaces](https://huggingface.co/spaces/ggunio/intelligent-tokenizer-v6-demo)
- 📝 **Documentation**: [English](paper_english.md) | [한국어](paper_korean.md)

## 💻 Installation

```bash
# Clone repository
git clone https://github.com/Woojjggun/intelligent-tokenizer.git
cd intelligent-tokenizer

# Install dependencies
pip install torch==2.1.0
pip install numpy==1.24.0
pip install tqdm
```

## 🚀 Quick Start

### Option 1: Try Live Demo
🎮 **[Try it now on Hugging Face Spaces](https://huggingface.co/spaces/ggunio/intelligent-tokenizer-v6-demo)** - No installation required!

### Option 2: Use in Code
```python
from huggingface_hub import hf_hub_download
from src.core.byte_tokenizer_v6 import ByteTokenizerV6
from core.boundary_aware_model import BoundaryAwareTokenizerModel

# Download model from Hugging Face
model_path = hf_hub_download(repo_id="ggunio/intelligent-tokenizer-v6",
                             filename="pytorch_model.bin")

# Initialize tokenizer and model
tokenizer = ByteTokenizerV6(max_seq_len=256)
model = BoundaryAwareTokenizerModel.from_pretrained(model_path)

# Process text
text = "Hello, World! 안녕하세요!"
tokens = tokenizer.encode(text)
embeddings = model.encode(tokens)
```

## 📁 Project Structure

```
intelligent-tokenizer/
├── core/
│   ├── unified_model.py          # Main model architecture
│   ├── boundary_aware_model.py   # UTF-8 boundary detection
│   └── train.py                   # Training pipeline
├── src/
│   └── core/
│       └── byte_tokenizer_v6.py  # Byte tokenizer
├── trainer/
│   ├── intelligent_loss.py       # Multi-objective loss
│   └── scheduler.py               # Learning rate scheduling
├── paper_english.md               # English paper
├── paper_korean.md                # Korean paper
└── README.md                      # This file
```

## 🧪 Testing

```python
# Run simple test
python simple_test.py

# Test reconstruction
python test_reconstruction.py

# Benchmark speed
python benchmark_speed.py
```

## 🎯 Use Cases

1. **LLM Communication Layer**: Direct embedding-based communication
2. **RAG Systems**: Semantic search without tokenization overhead
3. **Edge AI**: Lightweight deployment on mobile devices
4. **Multilingual NLP**: True zero-shot cross-lingual transfer

## 📈 Training

### Dataset
- **Flores-200**: 204 languages with parallel sentences
- **Training Time**: ~24 hours on RTX 4070
- **Epochs**: 22 (20 Korean-only, 2 multilingual)

### Reproduce Training
```bash
python core/train.py \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_epochs 22 \
    --device cuda
```

## 🤝 Contributing

We welcome contributions! This is a POC project and we plan to open-source based on community interest.

### Areas for Improvement
- [ ] Balanced multilingual training
- [ ] Chinese/Arabic performance enhancement
- [ ] Complex-valued embeddings research
- [ ] Adaptive window sizes for CJK
- [ ] Edge device optimization

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{woo2025attention,
  title={Attention Needs No Vocabulary: Byte-Level Learning for Universal Tokenization},
  author={Woo, Jinhyun},
  year={2025},
  url={https://github.com/Woojjggun/intelligent-tokenizer}
}
```

## 🙏 Acknowledgments

- **Vaswani et al. (2017)** - Title inspiration from "Attention Is All You Need"
- **Meta AI** - Flores-200 dataset
- **PyTorch Team** - Deep learning framework
- **Claude (Anthropic)** - Implementation assistance
- **Open-source Community** - Pioneering work in neural tokenization

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Jinhyun Woo** - ggunio5782@gmail.com

---

*"In the beginning was the Byte, and the Byte was with AI, and the Byte was AI."*

**Status**: 🚧 POC Stage - Not production ready  
**Goal**: Democratize AI through vocabulary-free tokenization  
**Vision**: Universal AI communication protocol without language barriers
