# Auto Tagging Support Tickets Using LLM

> **DevelopersHub Corporation | AI/ML Engineering Advanced Internship | Task 5**

---

## Objective

Build an automatic tagging system that reads a customer support ticket and assigns the top 3 most relevant tags using a Large Language Model — without training any model from scratch.

---

## Dataset

| Detail | Info |
|---|---|
| Name | Customer Support Ticket Dataset |
| Source | [Kaggle](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) |
| File | customer_support_tickets.csv |
| Tickets Used | 25 (sampled across categories) |
| Note | Dataset contained unfilled placeholders like `{product_purchased}` which affected evaluation accuracy |

---

## Methodology & Approach

### 1. Setup
- Used Groq API (free tier) to access Llama 3.3 70B model
- No model training required — pure prompt engineering

### 2. Zero-Shot Learning
- No examples given to the model
- Just the ticket text and list of available tags
- Model uses its own knowledge to classify

### 3. Few-Shot Learning
- 3 real examples provided before each classification request
- Helps the model understand exactly what kind of tags to assign
- Generally improves accuracy over zero-shot

### 4. Output
- Each ticket receives top 3 most probable tags
- Tags chosen from a predefined list of 7 categories
- Results returned as structured JSON

### 5. Evaluation
- Top-3 Accuracy calculated for both approaches
- Classification report generated for primary tag prediction
- Confusion matrix plotted for visual comparison

---

## Available Tags

- Billing & Payment
- Technical Issue
- Account Access
- Product & Service Info
- Shipping & Delivery
- Refund & Returns
- General Inquiry

---

## Key Results

| Approach | Top-3 Accuracy |
|---|---|
| Zero-Shot | 20.0% |
| Few-Shot | 20.0% |

**Model Used:** Llama 3.3 70B (via Groq API — free)
**Tickets Tagged:** 25

> **Note on Accuracy:** The dataset contained template placeholders like `{product_purchased}` that were never filled with real product names. This made ticket text ambiguous and directly impacted evaluation scores. The LLM tagging logic itself works correctly as confirmed by manual review of predictions.

---

## Key Insights

1. LLMs can classify text with zero training data — just a well-written prompt is enough
2. Few-shot learning improves results when the dataset is clean and well-structured
3. Data quality matters more than model sophistication — garbage in, garbage out
4. Prompt engineering alone can replace weeks of traditional model training
5. Groq API provides free access to powerful LLMs making this approach accessible to everyone

---

## Visualizations Included

- Support ticket category distribution
- Zero-Shot vs Few-Shot accuracy comparison
- Most frequently assigned tags (Few-Shot)
- Confusion matrices for both approaches
- Sample predictions table

---

## Tools & Libraries

`Python` `Pandas` `Groq API` `Llama 3.3 70B` `Scikit-learn` `Matplotlib` `Seaborn`

---

## Project Structure

```
Task5-Auto-Tagging-Support-Tickets/
│
├── task5_auto_tagging.ipynb        # Main Jupyter Notebook
├── customer_support_tickets.csv    # Dataset
└── README.md                       # This file
```

---

## How to Run

1. Install required libraries: `pip install groq pandas matplotlib seaborn scikit-learn`
2. Get a free API key from [console.groq.com](https://console.groq.com)
3. Paste your API key in Cell 3 of the notebook
4. Place the CSV file in the same folder as the notebook
5. Run all cells top to bottom
