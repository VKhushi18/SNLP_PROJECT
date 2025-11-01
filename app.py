import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import os
from typing import Dict, List, Tuple
import html
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Import visualization module
try:
    import visualization as vis
    import matplotlib.pyplot as plt
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False
    plt = None

# Try to import speech recognition libraries
try:
    import speech_recognition as sr
    import pyttsx3
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Improved language detection function
def detect_language_robust(text: str) -> str:
    """
    Robust language detection with multiple checks
    
    Args:
        text: Input text
        
    Returns:
        Detected language code
    """
    # Quick check: very short text is likely English
    if len(text.strip()) < 10:
        return 'en'
    
    # Use langdetect but with confidence check
    try:
        from langdetect import detect_langs
        
        # Get multiple language probabilities
        languages = detect_langs(text)
        
        # Check if English has high confidence
        for lang_info in languages:
            if lang_info.lang == 'en':
                # If English confidence is high enough, return English
                if lang_info.prob >= 0.7:
                    return 'en'
        
        # If no high-confidence English, return top language
        return languages[0].lang
        
    except Exception:
        # Fallback to simple detect
        try:
            return detect(text)
        except:
            return 'en'

# Language code to full name mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'pl': 'Polish',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'cs': 'Czech',
    'da': 'Danish',
    'fi': 'Finnish',
    'he': 'Hebrew',
    'hu': 'Hungarian',
    'no': 'Norwegian',
    'ro': 'Romanian',
    'uk': 'Ukrainian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'is': 'Icelandic',
    'ga': 'Irish',
    'mt': 'Maltese'
}

# Page configuration
st.set_page_config(
    page_title="CleanSpeak: AI Toxic Comment Detector",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful gradient theme
st.markdown("""
<style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
    }
    
    [data-testid="stSidebar"] section {
        background-color: transparent !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ecf0f1 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] label {
        color: #ecf0f1 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ecf0f1 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #ecf0f1 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox > div > div {
        color: #ecf0f1 !important;
        background-color: #3a556b !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        color: white !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Animated header */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Better contrast for headings */
    h1, h2, h3 {
        color: #1a1a1a !important;
    }
    
    /* Rounded cards */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e0eafc;
        padding: 15px;
        font-size: 1rem;
    }
    
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.7rem 2rem;
        border: none;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Custom metric cards */
    .toxicity-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .toxicity-card h4 {
        color: #333333;
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    
    .severity-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(to right, #ff6b6b, #feca57, #48dbfb);
        margin: 10px 0;
        display: flex;
        align-items: center;
        padding: 0 15px;
        color: white;
        font-weight: bold;
    }
    
    /* Highlight toxic words */
    .toxic-word {
        background-color: #ff6b6b;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* Tip box */
    .tip-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid #667eea;
    }
    
    /* Fix Streamlit info box contrast */
    .stAlert {
        color: #000000 !important;
    }
    
    [data-testid="stAlert"] {
        color: #000000 !important;
    }
    
    [data-testid="stAlert"] p, [data-testid="stAlert"] strong {
        color: #000000 !important;
    }
    
    /* Additional selectors for info boxes */
    .element-container [data-testid="stAlert"] div,
    .element-container [data-testid="stAlert"] p,
    .element-container [data-testid="stAlert"] li,
    .element-container [data-testid="stAlert"] strong {
        color: #000000 !important;
    }
    
    /* SHAP explanation styles */
    @keyframes fadeInWords {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .shap-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .shap-word {
        padding: 3px 6px;
        margin: 2px;
        border-radius: 4px;
        font-weight: 500;
        display: inline-block;
        animation: fadeInWords 0.5s ease-in;
        transition: all 0.3s ease;
    }
    
    .shap-word-positive {
        background-color: #ff6b6b;
        color: white;
    }
    
    .shap-word-positive:hover {
        background-color: #ff5252;
        transform: scale(1.05);
    }
    
    .shap-word-negative {
        background-color: #51cf66;
        color: white;
    }
    
    .shap-word-negative:hover {
        background-color: #40c057;
        transform: scale(1.05);
    }
    
    .shap-word-neutral {
        color: #000000;
    }
    
    /* Fix checkbox label visibility */
    [data-testid="stCheckbox"] label {
        color: #000000 !important;
    }
    
    .stCheckbox label {
        color: #000000 !important;
    }
    
    /* Fix all text element visibility */
    .element-container label,
    .element-container p,
    .element-container span {
        color: #000000 !important;
    }
    
    /* Fix all heading visibility */
    h3, h4, h5 {
        color: #000000 !important;
    }
    
    /* Fix tip box text */
    .tip-box, .tip-box * {
        color: #1a1a1a !important;
    }
    
    /* Fix markdown text */
    .stMarkdown {
        color: #000000 !important;
    }
    
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown p,
    .stMarkdown span {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Label definitions
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LABEL_EMOJIS = {
    'toxic': '☠️',
    'severe_toxic': '💀',
    'obscene': '🔞',
    'threat': '⚠️',
    'insult': '👊',
    'identity_hate': '🚫'
}

# Determine device
def get_device():
    """Get the appropriate device for inference"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Load model and tokenizer
@st.cache_resource
def load_model():
    """Load the locally trained DistilBERT model and tokenizer"""
    try:
        # Using our locally trained model
        model_path = "models/best_model"
        if not os.path.exists(model_path):
            # Fallback to Hugging Face model if local model doesn't exist
            model_name = "distilbert-base-uncased"
            st.info("Loading base model from Hugging Face...")
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(LABELS),
                problem_type="multi_label_classification"
            )
        else:
            st.success("Loading trained model from disk...")
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        # Move model to appropriate device
        device = get_device()
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load multilingual fallback model
@st.cache_resource
def load_multilingual_model():
    """Load the multilingual XLM-RoBERTa model for non-English text"""
    try:
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        device = get_device()
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.warning(f"Error loading multilingual model: {str(e)}")
        return None, None, None

# Load sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    """Load the RoBERTa sentiment analysis model"""
    try:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        device = get_device()
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.warning(f"Error loading sentiment model: {str(e)}")
        return None, None, None

# Preprocess text
def preprocess_text(text: str) -> str:
    """Clean and preprocess input text"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenize and encode text
def encode_text(tokenizer, text: str, max_length: int = 128):
    """Tokenize and encode text for model input"""
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']

# Predict toxicity
def predict_toxicity(model, tokenizer, text: str, device) -> Dict[str, float]:
    """Predict toxicity scores for all labels"""
    model.eval()
    
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Encode
    input_ids, attention_mask = encode_text(tokenizer, cleaned_text)
    
    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Create results dictionary
    results = {label: float(prob) for label, prob in zip(LABELS, probs)}
    return results

# Analyze sentiment
def analyze_sentiment(text: str) -> Dict[str, any]:
    """Analyze emotional sentiment of the text using RoBERTa sentiment model"""
    try:
        sentiment_tokenizer, sentiment_model, sentiment_device = load_sentiment_model()
        
        if sentiment_model is None:
            return None
        
        # Tokenize text
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(sentiment_device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        # Map to sentiment labels (cardiffnlp/twitter-roberta-base-sentiment labels)
        sentiment_labels = ['negative', 'neutral', 'positive']
        sentiment_probs = {sentiment_labels[i]: float(probs[i]) for i in range(len(sentiment_labels))}
        
        # Get dominant sentiment
        dominant_sentiment = max(sentiment_probs, key=sentiment_probs.get)
        confidence = sentiment_probs[dominant_sentiment]
        
        # Map sentiment to emotions
        emotion_mapping = {
            'negative': '😠 Angry/Disappointed',
            'neutral': '😐 Neutral',
            'positive': '😊 Happy/Positive'
        }
        emotion = emotion_mapping.get(dominant_sentiment, dominant_sentiment)
        
        return {
            "emotion": emotion,
            "sentiment": dominant_sentiment,
            "confidence": confidence,
            "all_scores": sentiment_probs
        }
        
    except Exception as e:
        st.warning(f"Sentiment analysis error: {str(e)}")
        return None

# Multilingual toxicity prediction
def predict_toxicity_multilingual(text: str, distilbert_tokenizer, distilbert_model, device) -> Dict[str, any]:
    """
    Multilingual toxicity prediction with automatic language detection and translation.
    
    Args:
        text: Input text in any language
        distilbert_tokenizer: Tokenizer for the DistilBERT model
        distilbert_model: Fine-tuned DistilBERT model
        device: Device for model inference
    
    Returns:
        Dictionary with language, translated_text, model_used, toxicity_label, and confidence
    """
    try:
        # Step 1: Detect language with robust detection
        detected_lang = detect_language_robust(text)
        
        # Initialize variables
        translated_text = None
        model_used = None
        toxicity_label = None
        confidence = 0.0
        
        # Step 2: Process based on detected language
        if detected_lang == 'en':
            # English text - use DistilBERT directly
            model_used = "DistilBERT"
            results = predict_toxicity(distilbert_model, distilbert_tokenizer, text, device)
            max_score = max(results.values())
            confidence = max_score
            toxicity_label = "Toxic" if max_score >= 0.5 else "Non-toxic"
            
            return {
                "language": detected_lang,
                "translated_text": None,
                "model_used": model_used,
                "toxicity_label": toxicity_label,
                "confidence": confidence,
                "all_scores": results
            }
        
        else:
            # Non-English text - translate to English
            try:
                translator = GoogleTranslator(source=detected_lang, target='en')
                translated_text = translator.translate(text)
                
                # Use DistilBERT on translated text
                model_used = "DistilBERT"
                results = predict_toxicity(distilbert_model, distilbert_tokenizer, translated_text, device)
                max_score = max(results.values())
                confidence = max_score
                toxicity_label = "Toxic" if max_score >= 0.5 else "Non-toxic"
                
                return {
                    "language": detected_lang,
                    "translated_text": translated_text,
                    "model_used": model_used,
                    "toxicity_label": toxicity_label,
                    "confidence": confidence,
                    "all_scores": results
                }
                
            except Exception as e:
                # Translation failed, use multilingual model as fallback
                st.warning(f"Translation failed: {str(e)}. Using multilingual model as fallback.")
                
                # Load multilingual model
                xlm_tokenizer, xlm_model, xlm_device = load_multilingual_model()
                
                if xlm_model is not None:
                    # Use XLM-RoBERTa for direct prediction
                    model_used = "XLM-RoBERTa"
                    encoding = xlm_tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(xlm_device)
                    attention_mask = encoding['attention_mask'].to(xlm_device)
                    
                    with torch.no_grad():
                        outputs = xlm_model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                    
                    # Map sentiment to toxicity (positive/neutral = non-toxic, negative = toxic)
                    confidence = probs[2] if len(probs) > 2 else probs[1]  # negative sentiment
                    toxicity_label = "Toxic" if confidence >= 0.5 else "Non-toxic"
                    
                    return {
                        "language": detected_lang,
                        "translated_text": None,
                        "model_used": model_used,
                        "toxicity_label": toxicity_label,
                        "confidence": confidence,
                        "all_scores": {}
                    }
                else:
                    # Fallback failed
                    return {
                        "language": detected_lang,
                        "translated_text": None,
                        "model_used": "Failed",
                        "toxicity_label": "Unknown",
                        "confidence": 0.0,
                        "all_scores": {}
                    }
                    
    except Exception as e:
        st.error(f"Multilingual prediction error: {str(e)}")
        return {
            "language": "unknown",
            "translated_text": None,
            "model_used": "Failed",
            "toxicity_label": "Unknown",
            "confidence": 0.0,
            "all_scores": {}
        }

# Highlight toxic words (simple keyword-based approach)
def highlight_toxic_words(text: str, toxic_keywords: List[str]) -> str:
    """Highlight potential toxic words in the text"""
    highlighted = text
    for word in toxic_keywords:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span class="toxic-word">{word}</span>',
            highlighted
        )
    return highlighted

# Get toxic keywords from prediction
def extract_toxic_keywords(text: str, results: Dict[str, float], threshold: float = 0.3) -> List[str]:
    """Extract words that might be contributing to toxicity"""
    # Common toxic keywords (can be enhanced with actual attention weights)
    toxic_keywords = [
        'stupid', 'idiot', 'moron', 'retard', 'fuck', 'shit', 'damn',
        'hate', 'kill', 'die', 'ass', 'bitch', 'whore', 'slut', 'bastard',
        'loser', 'pathetic', 'worthless', 'useless', 'disgusting'
    ]
    
    # Check if any toxicity detected
    max_toxicity = max(results.values())
    if max_toxicity > threshold:
        # Return matching words from the text
        found_words = [word for word in toxic_keywords if word in text.lower()]
        return found_words[:5]  # Return top 5
    return []

# Determine overall toxicity (Yes/No)
def get_toxicity_verdict(results: Dict[str, float], threshold: float = 0.5) -> Tuple[str, bool]:
    """Determine if comment is toxic (Yes/No)"""
    max_score = max(results.values())
    is_toxic = max_score >= threshold
    
    # Find which labels are active
    active_labels = [label for label, score in results.items() if score >= threshold]
    
    if is_toxic:
        if len(active_labels) == 0:
            verdict = "Yes (General Toxicity)"
        else:
            verdict = f"Yes - {', '.join([LABEL_EMOJIS.get(l, '') + ' ' + l.replace('_', ' ').title() for l in active_labels])}"
    else:
        verdict = "No"
    
    return verdict, is_toxic

def generate_shap_explanation(text, model, tokenizer, device):
    """Generate word importance using perturbation-based attribution"""
    try:
        model.eval()
        
        # Tokenize and get baseline prediction
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get baseline prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            baseline_prob = torch.max(probs).item()
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Compute importance by masking each token
        word_importances = {}
        for i in range(len(tokens)):
            token = tokens[i]
            
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '<pad>', '</s>', '<s>']:
                continue
            
            # Clean token
            clean_token = token.replace('▁', '').replace('##', '').strip()
            if not clean_token:
                continue
            
            # Create masked input
            masked_input_ids = input_ids.clone()
            masked_input_ids[0, i] = tokenizer.unk_token_id
            
            # Get prediction without this token
            with torch.no_grad():
                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                masked_prob = torch.max(probs).item()
            
            # Directional importance
            importance = baseline_prob - masked_prob
            
            # Accumulate importance (sum if word appears multiple times, but keep direction from dominant sign)
            if clean_token.lower() in word_importances:
                # Sum absolute values, preserve the dominant direction
                if abs(importance) > abs(word_importances[clean_token.lower()]):
                    # New importance is more significant, use it
                    word_importances[clean_token.lower()] = importance
                elif abs(importance) > 0.001:  # Only accumulate if significant
                    # Add but keep dominant sign
                    if importance * word_importances[clean_token.lower()] >= 0:
                        # Same direction, add
                        word_importances[clean_token.lower()] += importance
                    else:
                        # Different direction, take the one with larger magnitude
                        if abs(importance) > abs(word_importances[clean_token.lower()]):
                            word_importances[clean_token.lower()] = importance
            else:
                word_importances[clean_token.lower()] = importance
        
        return word_importances
        
    except Exception as e:
        st.warning(f"Explanation error: {str(e)}")
        return None

def visualize_shap_explanation(text, word_importances):
    """Create beautiful HTML visualization of word importance"""
    if word_importances is None or len(word_importances) == 0:
        return None
    
    try:
        # Normalize importance values to [-1, 1]
        importances_list = list(word_importances.values())
        if len(importances_list) == 0:
            return None
        
        max_importance = max(abs(min(importances_list)), abs(max(importances_list)))
        if max_importance > 0:
            word_importances_normalized = {k: v / max_importance for k, v in word_importances.items()}
        else:
            word_importances_normalized = word_importances
        
        # Get words and their importances
        words = text.split()
        highlighted_words = []
        
        # Higher threshold to show only significant contributions
        threshold = 0.15
        
        for word in words:
            word_clean = word.lower().strip('.,!?;:"()[]{}')
            
            # Check if this word or any substring has importance
            found_importance = None
            if word_clean in word_importances_normalized:
                found_importance = word_importances_normalized[word_clean]
            else:
                # Try substring matching for compound words
                for token, importance in word_importances_normalized.items():
                    if token in word_clean and len(token) >= 3:  # Only match if token is substantial
                        if found_importance is None or abs(importance) > abs(found_importance):
                            found_importance = importance
                        break
            
            # Apply threshold and highlight based on sign and magnitude
            if found_importance is not None and abs(found_importance) > threshold:
                # Positive importance = toxic (red), negative = decreases toxicity (green)
                if found_importance > 0:
                    highlighted_words.append(f'<span class="shap-word shap-word-positive">{html.escape(word)}</span>')
                else:
                    highlighted_words.append(f'<span class="shap-word shap-word-negative">{html.escape(word)}</span>')
            else:
                # Below threshold or no importance = neutral (no color)
                highlighted_words.append(f'<span class="shap-word shap-word-neutral">{html.escape(word)}</span>')
        
        html_content = f'''
        <div class="shap-container">
            <div style="font-size: 1.1rem; line-height: 1.8; color: #000000;">
                {" ".join(highlighted_words)}
            </div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; font-size: 0.9rem; color: #000000;">
                <strong>Legend:</strong> 
                <span style="background-color: #ff6b6b; color: white; padding: 2px 8px; border-radius: 3px;">🟥 Toxic (Increases)</span>
                <span style="background-color: #51cf66; color: white; padding: 2px 8px; border-radius: 3px; margin-left: 5px;">🟩 Neutral (Decreases)</span>
                <span style="color: #666; padding: 2px 8px; margin-left: 5px;">⚪ Weak/No Impact</span>
            </div>
        </div>
        '''
        
        return html_content
        
    except Exception as e:
        st.warning(f"Visualization error: {str(e)}")
        return None

# Voice Input Functions
@st.cache_resource
def get_speech_recognizer():
    """Initialize and cache the speech recognizer"""
    if not SPEECH_AVAILABLE:
        return None
    try:
        r = sr.Recognizer()
        return r
    except Exception as e:
        st.error(f"Failed to initialize speech recognizer: {str(e)}")
        return None

def speech_to_text():
    """Convert speech to text using microphone"""
    if not SPEECH_AVAILABLE:
        return None
    
    recognizer = get_speech_recognizer()
    if recognizer is None:
        return None
    
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Listen for audio
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        st.info("🔄 Processing audio...")
        # Recognize using Google Speech Recognition
        text = recognizer.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        st.warning("⏱️ No speech detected. Please try again.")
        return None
    except sr.UnknownValueError:
        st.warning("❓ Could not understand the audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"🔴 Speech recognition service error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"🔴 Error during speech recognition: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech"""
    if not SPEECH_AVAILABLE:
        return
    
    try:
        engine = pyttsx3.init()
        # Set speech properties
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level
        # Speak the text
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Text-to-speech unavailable: {str(e)}")

# Combined Voice Analysis Function
def voice_analysis_pipeline(spoken_text: str, distilbert_tokenizer, distilbert_model, device) -> Dict[str, any]:
    """
    Complete voice analysis pipeline:
    1. Convert speech to text (already done)
    2. Detect toxicity using multilingual pipeline
    3. Detect sentiment using RoBERTa
    4. Return structured results
    
    Args:
        spoken_text: Transcribed text from speech-to-text
        distilbert_tokenizer: Tokenizer for toxicity model
        distilbert_model: Toxicity model
        device: Device for inference
    
    Returns:
        Dictionary with toxicity and sentiment results
    """
    try:
        # Step 1: Already have spoken_text
        
        # Step 2: Detect language and get toxicity prediction
        detected_lang = detect_language_robust(spoken_text)
        
        # Get toxicity results
        if detected_lang != 'en':
            multilingual_result = predict_toxicity_multilingual(spoken_text, distilbert_tokenizer, distilbert_model, device)
            toxicity_label = multilingual_result.get('toxicity_label', 'Unknown')
            toxicity_confidence = multilingual_result.get('confidence', 0.0)
            text_for_sentiment = multilingual_result.get('translated_text') or spoken_text
        else:
            results = predict_toxicity(distilbert_model, distilbert_tokenizer, spoken_text, device)
            max_score = max(results.values())
            toxicity_confidence = max_score
            toxicity_label = "Toxic" if max_score >= 0.5 else "Non-toxic"
            text_for_sentiment = spoken_text
        
        # Step 3: Detect sentiment
        sentiment_result = analyze_sentiment(text_for_sentiment)
        if sentiment_result:
            # Extract just the emotion part (e.g., "Angry/Disappointed" from "😠 Angry/Disappointed")
            emotion_full = sentiment_result['emotion']
            sentiment_label = ' '.join(emotion_full.split()[1:]) if len(emotion_full.split()) > 1 else emotion_full
            sentiment_confidence = sentiment_result.get('confidence', 0.0)
        else:
            sentiment_label = "Unknown"
            sentiment_confidence = 0.0
        
        # Return structured results
        return {
            "input_text": spoken_text,
            "language": detected_lang,
            "toxicity_label": toxicity_label,
            "toxicity_confidence": toxicity_confidence,
            "sentiment_label": sentiment_label,
            "sentiment_confidence": sentiment_confidence,
            "success": True
        }
        
    except Exception as e:
        st.error(f"Voice analysis pipeline error: {str(e)}")
        return {
            "input_text": spoken_text,
            "toxicity_label": "Error",
            "toxicity_confidence": 0.0,
            "sentiment_label": "Error",
            "sentiment_confidence": 0.0,
            "success": False,
            "error": str(e)
        }

# Streamlit UI
def main():
    # Header
    st.markdown('<h1 class="main-header">💬 CleanSpeak: AI Toxic Comment Detector ⚡</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.title("📋 About CleanSpeak")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        page = st.selectbox("Choose a page:", ["🔍 Toxicity Detector", "📊 Dataset Visualization"])
        
        st.markdown("---")
        
        if page == "📊 Dataset Visualization":
            # Display visualization page
            if VIS_AVAILABLE:
                visualization_page()
            else:
                st.error("Visualization module not available. Please install matplotlib, seaborn, and wordcloud.")
            st.stop()
        
        st.markdown("""
        **CleanSpeak** is an AI-driven toxicity detector powered by BERT.
        
        ### Features:
        - 🔍 Real-time detection
        - 🧩 Multi-label classification
        - 🎨 Beautiful gradient UI
        - 💬 Word highlighting
        - ⚡ Fast & lightweight
        
        ### Toxicity Types:
        - **Toxic**: General toxicity
        - **Severe Toxic**: Extreme toxicity
        - **Obscene**: Profane language
        - **Threat**: Threatening language
        - **Insult**: Insulting content
        - **Identity Hate**: Hate speech
        """)
        
        st.markdown("---")
        st.markdown("**Model:** DistilBERT (Locally Trained)")
        st.markdown("**Device:** GPU (MPS) / CPU")
        
        if not SPEECH_AVAILABLE:
            st.info("💡 **Voice Features:** Install speechrecognition, pyttsx3, and pyaudio for voice input/output")
    
    # Load model
    tokenizer, model, device = load_model()
    
    if tokenizer is None or model is None or device is None:
        st.error("Failed to load the model. Please check your internet connection.")
        return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter a Comment to Analyze")
        
        # Text input with voice button
        col_input, col_voice = st.columns([5, 1])
        
        with col_input:
            # Initialize session state for voice input
            if 'voice_text' not in st.session_state:
                st.session_state['voice_text'] = ""
            
            user_input = st.text_area(
                "Type or paste a comment here:",
                value=st.session_state.get('voice_text', ''),
                placeholder="Example: This is a test comment...",
                height=150,
                label_visibility="collapsed"
            )
            # Clear voice_text after use
            if st.session_state.get('voice_text'):
                st.session_state['voice_text'] = ""
        
        with col_voice:
            if SPEECH_AVAILABLE:
                if st.button("🎤", use_container_width=True, help="Click to speak your comment"):
                    spoken_text = speech_to_text()
                    if spoken_text:
                        st.session_state['voice_text'] = spoken_text
                        st.rerun()
        
        # New: Voice Analysis button (auto-analyze after recording)
        if SPEECH_AVAILABLE:
            if st.button("🎤🔊 Voice Analysis (Speak & Auto-Analyze)", use_container_width=True):
                with st.spinner("🎤 Listening..."):
                    spoken_text = speech_to_text()
                    if spoken_text:
                        with st.spinner("🔍 Analyzing toxicity and sentiment..."):
                            # Run voice analysis pipeline
                            analysis_result = voice_analysis_pipeline(spoken_text, tokenizer, model, device)
                            
                            if analysis_result.get('success'):
                                # Display transcribed text
                                st.markdown("### 🎤 Recognized Speech")
                                st.text_area("Transcribed Text:", value=analysis_result['input_text'], height=100, key="transcribed", label_visibility="collapsed")
                                
                                # Display language if not English
                                if analysis_result.get('language') != 'en':
                                    lang_display = LANGUAGE_NAMES.get(analysis_result['language'], analysis_result['language'].upper())
                                    st.info(f"🌍 **Detected Language:** {lang_display} ({analysis_result['language'].upper()})")
                                
                                # Display Toxicity Result
                                st.markdown("### 🧠 Toxicity Analysis")
                                toxicity_label = analysis_result['toxicity_label']
                                toxicity_conf = analysis_result['toxicity_confidence']
                                
                                if toxicity_label == "Toxic":
                                    st.markdown(f"""
                                    <div style="background-color: #ffebee; border-left: 5px solid #f44336; padding: 20px; border-radius: 10px; margin: 10px 0;">
                                        <h3 style="color: #c62828; margin: 0;">🔴 Toxic</h3>
                                        <p style="color: #333; font-size: 1.2rem; margin: 5px 0;">Confidence: {toxicity_conf:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    voice_feedback = "Your message sounds harsh. Please reconsider."
                                else:
                                    st.markdown(f"""
                                    <div style="background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 20px; border-radius: 10px; margin: 10px 0;">
                                        <h3 style="color: #2e7d32; margin: 0;">🟢 Non-toxic</h3>
                                        <p style="color: #333; font-size: 1.2rem; margin: 5px 0;">Confidence: {toxicity_conf:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    voice_feedback = "Your message sounds kind and positive!"
                                
                                # Display Sentiment Result
                                st.markdown("### 💬 Sentiment Analysis")
                                sentiment_label = analysis_result['sentiment_label']
                                sentiment_conf = analysis_result['sentiment_confidence']
                                
                                # Determine color for sentiment
                                if "Positive" in sentiment_label or "Happy" in sentiment_label:
                                    sent_color = "#4caf50"
                                    sent_bg = "#e8f5e9"
                                    sent_emoji = "🟢"
                                elif "Negative" in sentiment_label or "Angry" in sentiment_label or "Disappointed" in sentiment_label:
                                    sent_color = "#f44336"
                                    sent_bg = "#ffebee"
                                    sent_emoji = "🔴"
                                else:  # Neutral
                                    sent_color = "#ff9800"
                                    sent_bg = "#fff3e0"
                                    sent_emoji = "🟡"
                                
                                st.markdown(f"""
                                <div style="background-color: {sent_bg}; border-left: 5px solid {sent_color}; padding: 20px; border-radius: 10px; margin: 10px 0;">
                                    <h3 style="color: {sent_color}; margin: 0;">{sent_emoji} {sentiment_label}</h3>
                                    <p style="color: #333; font-size: 1.2rem; margin: 5px 0;">Confidence: {sentiment_conf:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Voice feedback button
                                if st.button("🔊 Listen to Voice Feedback", use_container_width=True):
                                    text_to_speech(voice_feedback)
                            else:
                                st.error("Voice analysis failed. Please try again.")
        
        # Detect button
        if st.button("🔘 Detect Toxicity", use_container_width=True):
            if user_input.strip():
                with st.spinner("🔍 Analyzing toxicity..."):
                    # Detect language first with robust detection
                    detected_lang = detect_language_robust(user_input)
                    
                    # Use multilingual function if not English
                    text_for_explanation = user_input  # Default to original text
                    multilingual_result = None  # Initialize to track if translation happened
                    if detected_lang != 'en':
                        # Use multilingual prediction
                        multilingual_result = predict_toxicity_multilingual(user_input, tokenizer, model, device)
                        
                        # Display language info
                        lang_display = LANGUAGE_NAMES.get(detected_lang, detected_lang.upper())
                        st.info(f"🌍 **Detected Language:** {lang_display} ({detected_lang.upper()})")
                        
                        if multilingual_result.get('translated_text'):
                            st.info(f"📝 **Translation:** {multilingual_result['translated_text']}")
                            text_for_explanation = multilingual_result['translated_text']  # Use translated text for explanation
                        
                        st.info(f"🤖 **Model Used:** {multilingual_result['model_used']}")
                        
                        # Get results from multilingual prediction
                        results = multilingual_result.get('all_scores', {})
                        is_toxic = multilingual_result.get('toxicity_label') == 'Toxic'
                        
                        # Create verdict
                        if is_toxic:
                            # Get active labels from scores if available
                            if results:
                                active_labels = [label for label, score in results.items() if score >= 0.5]
                                if active_labels:
                                    verdict = f"Yes - {', '.join([LABEL_EMOJIS.get(l, '') + ' ' + l.replace('_', ' ').title() for l in active_labels])}"
                                else:
                                    verdict = "Yes (General Toxicity)"
                            else:
                                verdict = f"Yes (Confidence: {multilingual_result['confidence']:.2%})"
                        else:
                            verdict = "No"
                    else:
                        # English text - use regular prediction
                        results = predict_toxicity(model, tokenizer, user_input, device)
                        
                        # Get verdict
                        verdict, is_toxic = get_toxicity_verdict(results, threshold=0.5)
                    
                    # Display verdict prominently
                    if is_toxic:
                        st.error(f"### 🚨 Toxicity Detected: **{verdict}**")
                    else:
                        st.success(f"### ✅ Toxicity Status: **{verdict}**")
                    
                    # Sentiment Analysis
                    st.markdown("### 💭 Emotional Sentiment Analysis")
                    with st.spinner("Analyzing emotions..."):
                        try:
                            sentiment_result = analyze_sentiment(text_for_explanation)
                            if sentiment_result:
                                # Display sentiment
                                sentiment_cols = st.columns(3)
                                with sentiment_cols[0]:
                                    st.markdown(f"""
                                    <div style="background: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #667eea;">
                                        <h4 style="color: #333; font-weight: 600; margin-bottom: 10px;">{sentiment_result['emotion']}</h4>
                                        <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{sentiment_result['confidence']:.1%}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Display all sentiment scores
                                with sentiment_cols[1]:
                                    for sentiment, score in sentiment_result['all_scores'].items():
                                        emoji = '😠' if sentiment == 'negative' else '😐' if sentiment == 'neutral' else '😊'
                                        st.metric(f"{emoji} {sentiment.capitalize()}", f"{score:.1%}")
                        except Exception as e:
                            st.warning("Sentiment analysis unavailable")
                    
                    # Voice feedback button
                    if SPEECH_AVAILABLE:
                        feedback_cols = st.columns(3)
                        with feedback_cols[0]:
                            if st.button("🔊 Listen to Feedback", use_container_width=True):
                                feedback_text = f"Toxicity status: {verdict}. "
                                if sentiment_result and sentiment_result.get('emotion'):
                                    feedback_text += f"Emotional sentiment: {sentiment_result['emotion'].split()[1] if len(sentiment_result['emotion'].split()) > 1 else sentiment_result['emotion']}. "
                                text_to_speech(feedback_text)
                    
                    # Display detailed results
                    st.markdown("### 📊 Detailed Toxicity Breakdown")
                    
                    # Create columns for metrics
                    cols = st.columns(2)
                    
                    for idx, (label, score) in enumerate(results.items()):
                        col_idx = idx % 2
                        emoji = LABEL_EMOJIS.get(label, '📌')
                        display_name = label.replace('_', ' ').title()
                        
                        with cols[col_idx]:
                            # Custom metric card
                            st.markdown(f"""
                            <div class="toxicity-card">
                                <h4>{emoji} {display_name}</h4>
                                <div class="severity-bar" style="width: {score*100}%">
                                    {score*100:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # SHAP Explanation - always show directly under results (use translated text for non-English)
                    with st.spinner("🔍 Generating word importance explanation..."):
                        try:
                            word_importances = generate_shap_explanation(text_for_explanation, model, tokenizer, device)
                            if word_importances is not None:
                                shap_html = visualize_shap_explanation(text_for_explanation, word_importances)
                                if shap_html:
                                    st.markdown("### 🎯 Word Importance Explanation")
                                    # Add note for non-English texts
                                    if detected_lang != 'en' and multilingual_result and multilingual_result.get('translated_text'):
                                        st.info("ℹ️ *Word importance analysis is based on the English translation above.*")
                                    st.markdown(shap_html, unsafe_allow_html=True)
                                else:
                                    st.info("Could not generate visualization.")
                            else:
                                st.warning("Explanation generation failed.")
                        except Exception as e:
                            st.error(f"Error generating explanation: {str(e)}")
                    
                    # Tip
                    if is_toxic:
                        st.markdown("""
                        <div class="tip-box">
                            ✅ <strong>Tip:</strong> Try rephrasing harsh words for a kinder comment :)
                        </div>
                        """, unsafe_allow_html=True)
                    
            else:
                st.warning("⚠️ Please enter a comment to analyze.")
    
    with col2:
        st.markdown("### 🎯 Quick Stats")
        st.info("""
        **Analyze any comment:**
        1. Type or paste text
        2. Click Detect
        3. Get instant results
        
        **The model checks for:**
        - General toxicity
        - Severe toxicity
        - Obscenity
        - Threats
        - Insults
        - Identity-based hate
        """)
        
        st.markdown("---")
        
        # Example button
        if st.button("📌 Try Example", use_container_width=True):
            example = "I love this product! It works amazingly well."
            st.session_state['example_text'] = example
            st.rerun()

def visualization_page():
    """Display the dataset visualization page"""
    st.markdown('<h1 class="main-header">📊 Dataset Visualization Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Dataset Analysis Dashboard! 📈
    
    This page provides insights into the Jigsaw Toxic Comment Classification dataset.
    Explore the distribution of toxicity types, word clouds, and label overlaps.
    """)
    
    # Check if train.csv exists
    if not os.path.exists('train.csv'):
        st.error("❌ train.csv not found in the current directory.")
        st.info("💡 Please make sure train.csv is in the same directory as app.py")
        return
    
    # Generate visualizations
    with st.spinner("🎨 Generating visualizations..."):
        try:
            fig1, fig2, fig3, fig4, fig5 = vis.main_visualization('train.csv')
        except Exception as e:
            st.error(f"Error generating visualizations: {str(e)}")
            return
    
    if fig1 is None:
        st.error("Failed to generate visualizations")
        return
    
    # Display visualizations
    st.markdown("---")
    
    # Label frequency chart
    st.markdown("### 📊 Label Distribution")
    st.markdown("This chart shows how many comments belong to each toxicity category.")
    st.pyplot(fig1)
    plt.close(fig1)
    
    # Pie chart
    st.markdown("### 🧩 Toxic vs Non-Toxic Distribution")
    st.markdown("Overall distribution of toxic and non-toxic comments in the dataset.")
    st.pyplot(fig4)
    plt.close(fig4)
    
    # Heatmap
    st.markdown("### 🔥 Label Co-occurrence Heatmap")
    st.markdown("This heatmap shows which toxicity labels often appear together in the same comment.")
    st.pyplot(fig5)
    plt.close(fig5)
    
    # Word clouds
    st.markdown("---")
    st.markdown("### 💬 Word Clouds")
    st.markdown("""
    **Word clouds** visualize the most frequent words in toxic vs non-toxic comments.
    Larger words appear more frequently in that category.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Toxic Comments")
        st.pyplot(fig2)
        plt.close(fig2)
    
    with col2:
        st.markdown("#### 🟢 Non-Toxic Comments")
        st.pyplot(fig3)
        plt.close(fig3)
    
    # Regenerate button
    st.markdown("---")
    if st.button("🔄 Regenerate All Visualizations", use_container_width=True):
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>📚 <strong>Dataset:</strong> Jigsaw Toxic Comment Classification Challenge</p>
        <p>🛠️ <strong>Tools:</strong> Pandas, Matplotlib, Seaborn, WordCloud</p>
    </div>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize session state"""
    if 'example_text' not in st.session_state:
        st.session_state.example_text = ""

if __name__ == "__main__":
    init_session_state()
    main()

