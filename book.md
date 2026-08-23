---
title: "Introduction to Modern AI"
subtitle: "A Hands-On Curriculum for Curious Self-Learners"
author: "SaraKimsLabs"
date: "2026"
geometry: "margin=1in"
fontsize: "11pt"
linestretch: 1.2
toc: true
toc-depth: 2
---

# Preface

Welcome to **Introduction to Modern AI**. This curriculum is designed to take you from core computational mental models to building practical modern AI systems—without skipping the mechanics or getting bogged down in unapplied theory.

---

# Module 01: The AI Landscape Decoded

## 1. Key Terminology
* **Artificial Intelligence (AI):** Systems that perform tasks requiring human-like intelligence.
* **Machine Learning (ML):** Systems that extract patterns directly from data rather than hardcoded rules.
* **Deep Learning (DL):** Artificial neural networks learning hierarchical feature representations.
* **Generative AI (GenAI):** Models producing novel artifacts (text, code, images) from prompts.

## 2. The Core Paradigm Shift
Traditional software executes human-crafted rules on input data:
$$\text{Rules} + \text{Data} \rightarrow \text{Answers}$$

Machine learning infers the mapping function from historical examples:
$$\text{Data} + \text{Historical Answers} \rightarrow \text{Learned Model}$$

---

# Module 02: How Models Learn

## 1. Key Mechanics
* **Weights ($w$) & Bias ($b$):** The adjustable parameters of the model.
* **Loss Function:** A metric scoring prediction error.
* **Gradient Descent:** Tweaking weights in the direction of steepest loss descent.
* **Learning Rate ($\alpha$):** The step size taken downhill during parameter updates.

## 2. Optimization Loop
Every supervised model follows a continuous 4-step loop:
1. **Forward Pass:** Predict output using current parameters ($y = wx + b$).
2. **Compute Loss:** Calculate error against ground truth.
3. **Compute Gradients:** Evaluate partial derivatives ($\frac{\partial \text{Loss}}{\partial w}$, $\frac{\partial \text{Loss}}{\partial b}$).
4. **Update Parameters:** Adjust weights: $w \leftarrow w - \alpha \cdot \frac{\partial \text{Loss}}{\partial w}$.

---

# Module 03: Neural Networks & Deep Learning

## 1. Multi-Layer Representations
Linear models are restricted to linear hyperplanes. By inserting non-linear activation functions (e.g., ReLU: $f(x) = \max(0, x)$) between matrix multiplications, neural networks can approximate arbitrary continuous functions.

## 2. Backpropagation
Using the calculus chain rule, backpropagation distributes loss gradients from output layers backward through all intermediate hidden layers to adjust internal weights.

---

# Module 04: Language & Vector Embeddings

## 1. Turning Meaning into Geometry
Computers process numerical arrays rather than raw text. Tokenizers convert characters into integers, and embedding layers project token sequences into continuous dense vector spaces ($\mathbb{R}^d$).

## 2. Semantic Proximity
Semantic similarity is calculated by evaluating the angle between vectors via **Cosine Similarity**:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

---

# Module 05: The Transformer & Generative AI

## 1. Self-Attention Mechanics
Unlike sequential Recurrent Neural Networks (RNNs), Transformers compute attention weights across all tokens in parallel:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This allows the model to capture long-range contextual relationships dynamically.

## 2. Sampling Controls
* **Temperature:** Adjusts the sharpness of the softmax probability distribution.
* **Top-p (Nucleus Sampling):** Restricts token choices to the top cumulative probability mass $p$.

---

# Module 06: RAG & Practical AI Systems

## 1. Overcoming Knowledge Cutoffs & Hallucinations
Retrieval-Augmented Generation (RAG) grounds language models on external knowledge bases:
1. **Chunk & Embed:** Slice internal documents into semantic chunks and store them in a vector index.
2. **Retrieve:** Match user queries against chunk vectors using nearest-neighbor search.
3. **Augment & Generate:** Inject retrieved passages into the prompt context window.

---

# Module 07: Ethics, Limits & Next Frontiers

## 1. Deployment Defenses
Production AI systems require boundary checks:
* **Input Moderation:** Regex and semantic classifiers to block prompt injections and jailbreaks.
* **Output Sanitization:** Redaction of Personally Identifiable Information (PII) and compliance filters.

---

# Course Capstone Project

To complete the course, assemble an end-to-end grounded assistant:
1. Embed a custom document collection using dense vector representations.
2. Query the vector index to retrieve relevant passages.
3. Pass context into a local language model for grounded inference.
4. Wrap inputs and outputs with security guardrails and PII redaction.
