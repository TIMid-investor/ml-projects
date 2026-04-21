# ML Projects: Interview-Ready Portfolio

**Goal**: 4 intentionally different projects = ~90% interview coverage

Build credibility in tabular ML, NLP pipelines, deep learning, and interpretability. Each project teaches a distinct skill set that interviewers actually care about.

---

## 📐 Project Structure

```
projects/ml/
├── 1_market_regime_model/      # Tabular ML + Probabilistic Thinking
├── 2_earnings_sentiment/        # NLP + ML Pipeline
├── 3_volatility_forecaster/     # PyTorch + Sequence Models
├── 4_macro_market_model/        # Interpretability + SHAP
├── shared/                      # Common utilities, data loaders
└── notebooks/                   # Exploration & analysis
```

---

## 🎯 Project Breakdown

### 1️⃣ Market Regime Model
**Category**: Tabular ML + Probabilistic Thinking

**Goal**: Classify market regime (bull/sideways/bear) from price and macro data

**Tools**:
- scikit-learn (baseline, preprocessing)
- XGBoost (main model)

**What You'll Learn**:
- Feature engineering (HUGE interview topic)
- Classification metrics beyond accuracy (precision, recall, ROC-AUC)
- Time-series cross-validation (avoiding look-ahead bias)
- Overfitting in finance models
- Probabilistic output calibration

**Why This Stays**:
- Your core "finance + ML" credibility project
- Most accessible entry point
- Foundation for later projects

**Deliverables**:
- `market_regime_model.py` - Training pipeline
- `regime_features.py` - Feature engineering module
- `notebooks/01_regime_exploration.ipynb` - Analysis & results
- Model evaluation report with confusion matrix, ROC curves

---

### 2️⃣ Earnings Sentiment Pipeline
**Category**: NLP + ML Pipeline

**Goal**: Extract sentiment from earnings call transcripts → predict next quarter returns

**Tools**:
- MLX + Qwen (LLM for sentiment extraction)
- scikit-learn (downstream classifier)
- Optional: PyTorch (if fine-tuning)

**What You'll Learn**:
- Unstructured → structured data pipelines
- Feature extraction from text (LLM-based vs traditional NLP)
- Labeling challenges & domain expertise
- Data quality at scale
- When to use LLMs vs simpler methods

**Why This Matters**:
- Most finance candidates lack NLP exposure
- Shows you understand modern LLM workflows
- Demonstrates pipeline thinking

**Deliverables**:
- `earnings_pipeline.py` - End-to-end pipeline
- `sentiment_extractor.py` - LLM-based feature extraction
- `notebooks/02_sentiment_analysis.ipynb` - EDA + model results
- Sentiment validation & distribution analysis

---

### 3️⃣ Volatility Forecaster
**Category**: Deep Learning + When NOT to Use It

**Goal**: Predict 1-5 day ahead volatility using LSTM or transformer

**Tools**:
- PyTorch (LSTM or simple neural net)
- Tensorboard (training visualization)

**What You'll Learn**:
- Tensors & tensor operations
- Training loops & backprop intuition
- Loss functions for regression
- Batch processing & normalization
- **Most Important**: Why deep learning often fails in markets
  - Overfitting to noise
  - Limited data vs model complexity
  - When simpler methods outperform

**Why You Need This**:
- Without a PyTorch project, you sound theoretical
- Your real insight: "I've tried this, here's why it didn't work"
- Shows honest evaluation of tools

**Deliverables**:
- `volatility_lstm.py` - PyTorch model architecture
- `volatility_trainer.py` - Training loop with validation
- `notebooks/03_volatility_deep_dive.ipynb` - Results & failure analysis
- Comparative analysis: LSTM vs XGBoost baseline

---

### 4️⃣ Macro → Market Model
**Category**: Interpretability + Business Thinking

**Goal**: Predict market direction from macro indicators (GDP, inflation, yield curve) with explainability

**Tools**:
- XGBoost (interpretable baseline)
- scikit-learn (preprocessing, ensemble)
- SHAP (feature importance & explanations)

**What You'll Learn**:
- Model interpretability (SHAP values, feature importance)
- Nonlinear relationships in macro data
- Scenario analysis & stress testing
- How to explain models to non-technical stakeholders
- Actionable insights from complex models

**Why This Matters**:
- Your "I can tell the story" project
- Interviewers want someone who explains results, not just trains models
- SHAP is increasingly expected in ML interviews

**Deliverables**:
- `macro_market_model.py` - XGBoost with feature selection
- `explainability.py` - SHAP analysis & visualization
- `notebooks/04_macro_interpretability.ipynb` - Full analysis
- SHAP summary plots, force plots, decision plots

---

## 🛠️ Shared Utilities

**`shared/`** directory contains:
- `data_loader.py` - Download & preprocess market data
- `feature_utils.py` - Common feature engineering functions
- `metrics.py` - Custom evaluation metrics for finance
- `validation.py` - Time-series CV utilities

---

## 🚀 Getting Started

### Data Sources
- Market data: yfinance, Alpha Vantage, or local warehouse
- Earnings transcripts: Seeking Alpha, company IRs, or LLM-generated
- Macro data: FRED, World Bank, or public APIs

### Development Setup
```bash
cd ~/wealth-platform/projects/ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Workflow
1. **Explore** - Start in `notebooks/` with EDA
2. **Build** - Develop in project directories
3. **Evaluate** - Test on holdout data with time-series splits
4. **Document** - Update README + add results to notebooks

---

## 📊 Interview Talking Points

| Project | Key Phrase | Why It Matters |
|---------|-----------|---|
| Market Regime | "Feature engineering reduced overfitting by..." | Shows understanding of ML fundamentals |
| Earnings Sentiment | "We went from transcripts → labels → predictions in one pipeline" | Demonstrates full-stack thinking |
| Volatility Forecaster | "LSTM overfit; XGBoost outperformed by 40%" | Honest evaluation of tools |
| Macro Model | "SHAP shows yield curve is the #1 predictor, which aligns with..." | Connects models to domain knowledge |

---

## 📈 Success Criteria

- ✅ Each project has reproducible code + documented results
- ✅ Can explain why you chose each tool for each project
- ✅ Comparative analysis showing trade-offs (simple vs complex)
- ✅ Real data, real results (even if modest performance)
- ✅ At least one project fails gracefully (with clear post-mortem)

---

## 📚 Resources

**Tabular ML**:
- Feature engineering: Kaggle competitions, domain knowledge
- XGBoost: Official docs + Statquest videos

**NLP + LLMs**:
- MLX: Apple's framework for efficient LLM inference
- SHAP for text: Interpretation of LLM outputs

**PyTorch**:
- Official tutorials + fastai course
- Understand when deep learning fails (crucial)

**Interpretability**:
- SHAP documentation
- "Interpretable ML" book by Christoph Molnar

---

## 🔄 Version Control

Each project is self-contained:
```
git add 1_market_regime_model/
git commit -m "feat: market regime model with XGBoost baseline"
```

This keeps history clean and makes each project a discrete resume line.
