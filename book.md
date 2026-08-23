---
title: "Introduction to Modern AI"
subtitle: "A Hands-On Curriculum for Curious Self-Learners"
author: "SaraKimsLabs"
date: "2026"
geometry: "margin=1in"
fontsize: "11pt"
linestretch: 1.25
toc: true
toc-depth: 2
numbersections: true
---

\newpage

# Preface {-}

Welcome to **Introduction to Modern AI**. This curriculum is designed to take you from foundational computational mental models to building practical modern AI systems—without skipping the mechanics or getting bogged down in unapplied theory.

Each chapter covers theoretical intuition, mathematical formulation, and a self-contained Python implementation.

\newpage

# Module 01: The AI Landscape Decoded

## Key Terminology
* **Artificial Intelligence (AI):** Computational systems performing tasks that typically require human intelligence.
* **Machine Learning (ML):** Algorithms that extract statistical patterns directly from data rather than relying on static rules.
* **Deep Learning (DL):** Multi-layered artificial neural networks capable of learning hierarchical feature representations.
* **Generative AI (GenAI):** Probabilistic models trained to generate novel text, code, audio, or images from contextual prompts.

## The Core Paradigm Shift
Traditional software executes human-crafted rules on structured inputs:

$$\text{Rules} + \text{Data} \longrightarrow \text{Answers}$$

Machine learning infers the underlying mathematical mapping directly from historical examples:

$$\text{Data} + \text{Historical Answers} \longrightarrow \text{Learned Model}$$

## Hands-On Lab: Rule-Based vs. Machine Learning Classifier

```python
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

train_emails = [
    "Exclusive deal! Claim your free cash prize now",
    "URGENT: Your account requires immediate verification and money transfer",
    "Win a brand new luxury car today only",
    "Earn easy cash working from home fast",
    "Hey, are we still meeting for lunch at noon tomorrow?",
    "Project roadmap review notes attached for your feedback",
    "Can you review the latest quarterly earnings draft by Friday?",
    "Dinner tonight at 7 pm with the team?"
]
train_labels = [1, 1, 1, 1, 0, 0, 0, 0]

test_emails = [
    "Claim your free prize today",
    "Cash bonuses available for lunch meeting team",
    "Urgent: Send money to unlock your reward",
    "Can we reschedule our Friday review call?"
]
true_test_labels = [1, 0, 1, 0]

# Approach A: Rule-Based Classifier
SPAM_KEYWORDS = ["free", "cash", "prize", "urgent", "money", "win"]

def rule_based_classifier(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    for word in tokens:
        if word in SPAM_KEYWORDS:
            return 1
    return 0

rule_preds = [rule_based_classifier(email) for email in test_emails]
print(f"Rule-Based Accuracy: {accuracy_score(true_test_labels, rule_preds) * 100:.1f}%")

# Approach B: Statistical Machine Learning (Naive Bayes)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_emails)
X_test = vectorizer.transform(test_emails)

ml_model = MultinomialNB()
ml_model.fit(X_train, train_labels)

ml_preds = ml_model.predict(X_test)
print(f"Machine Learning Accuracy: {accuracy_score(true_test_labels, ml_preds) * 100:.1f}%")

