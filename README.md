# 💬 CleanSpeak: AI Toxic Comment Detector

CleanSpeak is an AI-driven web application designed to detect and classify toxic comments in real-time. Using a fine-tuned DistilBERT transformer on the Jigsaw Wikipedia Talk Page dataset, it accurately identifies multiple forms of online toxicity including hate speech, insults, threats, and obscenity.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.1-orange.svg)
![Transformers](https://img.shields.io/badge/transformers-4.36.0-green.svg)

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Real-Time Detection** | Detects and classifies toxicity as the user types. |
| 🧩 **Multi-label Classification** | Identifies multiple types of toxicity in one comment. |
| 🎨 **Aesthetic UI** | Gradient theme, animated cards, and severity bars. |
| 💬 **Word Highlighting** | Highlights words that contribute most to toxicity. |
| 📊 **Visual Dashboard** | Shows toxicity distribution and confidence bars. |
| ⚡ **Lightweight** | Uses DistilBERT → minimal training time, fast inference. |
| ☁️ **Deployed via Streamlit** | Easy to share and present live in college. |

## 🎯 Toxicity Classification

The model detects **6 types of toxicity**:

- **Toxic** ☠️: General toxicity
- **Severe Toxic** 💀: Extreme toxicity
- **Obscene** 🔞: Profane language
- **Threat** ⚠️: Threatening language
- **Insult** 👊: Insulting content
- **Identity Hate** 🚫: Hate speech

**Output Format**: **Yes** or **No** (with detailed breakdown)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cleanspeak-toxicity-detector.git
cd cleanspeak-toxicity-detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🏗️ Model Architecture

### Preprocessing
1. Lowercase all text
2. Remove unwanted symbols, URLs, HTML tags
3. Tokenize using DistilBERT tokenizer
4. Encode text → Input IDs + Attention Masks

### Model
- **Base Model**: distilbert-base-uncased (Hugging Face)
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: AdamW

###This repository contains a custom fine-tuned transformer model for toxic comment detection.
The model was trained from scratch using the Jigsaw Toxic Comment Classification dataset and is designed to detect multiple forms of online toxicity.

## 📊 Evaluation Metrics
- Accuracy per label
- Macro F1-score
- ROC-AUC per class
- Multi-label classification support

## 🎨 UI Features

### Theme
- Soft gradient background (#f8f9fa → #e0eafc)
- Animated header with smooth transitions
- Rounded, shadowed cards
- Colorful progress bars for each toxicity type

### Layout
```
----------------------------------------------------------
💬 CleanSpeak: AI Toxic Comment Detector
----------------------------------------------------------
[ Textbox: "Type or paste a comment..." ]

🔘 Detect Toxicity  → (Button)

[ Output Panel ]
----------------------------------------------------------
🚨 Toxicity Detected: Yes - ☠️ Toxic, 👊 Insult

📊 Detailed Toxicity Breakdown:
Toxic: ██████░░░░░░░░░░░░░░░░░ 60%
Severe Toxic: ██░░░░░░░░░░░░░░░ 20%
Obscene: ████░░░░░░░░░░░░░░░░ 40%
Threat: ░░░░░░░░░░░░░░░░░░░░░ 5%
Insult: ███████░░░░░░░░░░░░░░ 70%
Identity Hate: ██░░░░░░░░░░░░░ 10%

💬 Toxic Words: "stupid", "idiot"
----------------------------------------------------------
✅ Tip: Try rephrasing harsh words for a kinder comment :)
----------------------------------------------------------
```

## 🔧 Technical Details

### Dependencies
- `streamlit==1.29.0`: Web framework
- `transformers==4.36.0`: Hugging Face transformers
- `torch==2.1.1`: PyTorch backend
- `numpy==1.26.2`: Numerical operations
- `pandas==2.1.4`: Data manipulation
- `scikit-learn==1.3.2`: Evaluation metrics


## 📈 Use Cases

1. **Chat Moderation**: Real-time filtering of toxic messages
2. **Educational Platforms**: Promoting healthy digital communication
3. **Social Media**: Content moderation
4. **Research**: Toxicity analysis and sentiment detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Jigsaw** for the Wikipedia Toxic Comment Classification dataset
- **Hugging Face** for the transformers library and pre-trained models
- **Streamlit** for the amazing web framework
- **Research team** at Unitary AI for the toxic-bert model

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made for promoting healthier digital communication**

