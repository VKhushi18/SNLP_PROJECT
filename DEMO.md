# 🎥 CleanSpeak Demo Guide

## Quick Demo Script

### Scenario 1: Non-Toxic Comment
**Input:**
```
I love this product! It works amazingly well and has helped me tremendously.
```

**Expected Output:**
```
✅ Toxicity Status: No

All toxicity scores should be below 50%
```

---

### Scenario 2: Toxic Comment (Insults)
**Input:**
```
You are such a stupid idiot. This is the worst thing I've ever seen!
```

**Expected Output:**
```
🚨 Toxicity Detected: Yes - ☠️ Toxic, 👊 Insult

- Toxic: High score (70-90%)
- Insult: High score (70-90%)
- Other categories: Lower scores
- Toxic Words Highlighted: "stupid", "idiot"
```

---

### Scenario 3: Severe Toxicity
**Input:**
```
I hate you so much! You should just die already. Worthless piece of garbage.
```

**Expected Output:**
```
🚨 Toxicity Detected: Yes - ☠️ Toxic, 💀 Severe Toxic, 👊 Insult

- Toxic: Very high (80-95%)
- Severe Toxic: High (60-80%)
- Insult: High (70-85%)
- Toxic Words Highlighted: "hate", "die", "worthless"
```

---

### Scenario 4: Threatening Language
**Input:**
```
Watch your back. I'm coming for you.
```

**Expected Output:**
```
🚨 Toxicity Detected: Yes - ⚠️ Threat

- Threat: Moderate to high (50-70%)
- Other categories: Lower scores
```

---

### Scenario 5: Obscene Content
**Input:**
```
What the f*** is wrong with you? This is complete bull***t!
```

**Expected Output:**
```
🚨 Toxicity Detected: Yes - ☠️ Toxic, 🔞 Obscene

- Toxic: High (60-80%)
- Obscene: High (70-85%)
- Toxic Words Highlighted: (may vary based on filters)
```

---

### Scenario 6: Identity Hate
**Input:**
```
All [group] people are terrible and should not be allowed anywhere.
```

**Expected Output:**
```
🚨 Toxicity Detected: Yes - ☠️ Toxic, 🚫 Identity Hate

- Toxic: Moderate (50-70%)
- Identity Hate: Moderate to high (40-70%)
- Threat: May have some score
```

---

## 🎯 Testing Tips

1. **Start with clean examples** to see the "No" result
2. **Gradually increase toxicity** to see thresholds activate
3. **Try different combinations** of toxicity types
4. **Test edge cases** like sarcasm or context-dependent language
5. **Check word highlighting** to see which words trigger detection

## 📊 Understanding the Results

- **Yes**: At least one toxicity type scored ≥ 50%
- **No**: All toxicity types scored < 50%
- **Progress bars**: Visual representation of confidence scores
- **Toxic Words**: Keywords that likely contributed to the detection

## 🎨 UI Elements to Highlight

1. **Gradient background**: Shows on load
2. **Animated header**: Fade-in effect
3. **Color-coded results**: Red for Yes, Green for No
4. **Severity bars**: Gradient from red to yellow to blue
5. **Toxic word highlighting**: Red background with bold white text
6. **Tip box**: Appears when toxicity is detected

