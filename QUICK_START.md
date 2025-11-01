# ⚡ CleanSpeak Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Setup (One-time)
```bash
# Option A: Use setup script
./setup.sh

# Option B: Manual setup
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Open in Browser
Your browser should open automatically to `http://localhost:8501`

If not, manually navigate to that URL.

---

## 🎯 How to Use

1. **Type** a comment in the text box
2. **Click** "🔘 Detect Toxicity"
3. **See** the result:
   - ✅ **No** = Safe comment
   - 🚨 **Yes** = Toxic comment detected

---

## 📋 Key Files

- `app.py` - Main application
- `requirements.txt` - Dependencies
- `README.md` - Full documentation
- `DEMO.md` - Demo examples
- `setup.sh` - Automated setup script

---

## 🐛 Troubleshooting

### Issue: "No module named 'streamlit'"
**Solution:** Activate your virtual environment
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Model download is slow
**Solution:** Model downloads automatically on first run (~500MB)
- First run will take a few minutes
- Subsequent runs are instant

### Issue: "Port 8501 already in use"
**Solution:** Kill the existing process or use a different port
```bash
# Kill existing process
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8502
```

---

## 💡 Quick Tips

- **First run**: Model downloads from Hugging Face (~2-3 minutes)
- **Subsequent runs**: Instant startup
- **No training needed**: Uses pre-trained model
- **Yes/No output**: Simple binary classification
- **Detailed breakdown**: See individual toxicity types

---

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. See `DEMO.md` for example inputs/outputs
3. Open an issue on GitHub

---

**Happy detecting! 💬✨**

