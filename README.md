# Hands-On Notebooks 📓🧑‍💻

[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/)
[![Difficulty](https://img.shields.io/badge/difficulty-beginner-success)](#)
[![Duration](https://img.shields.io/badge/duration-3%20hours-brightgreen)](#)
![Docker + Notebook Test](https://github.com/BridgingAISocietySummerSchools/Hands-On-Notebooks/actions/workflows/build_run_test.yml/badge.svg)


A curated collection of **Jupyter notebooks** to explore and teach the fundamentals of **machine learning**. These notebooks are practical, beginner-friendly, and support our interdisciplinary summer school curriculum.


## 🚀 Quick Start

You can run all notebooks in your browser via **Google Colab** — no installation required:
👉 [Open in Google Colab](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/)

1. Test Notebook [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/01_test_notebook.ipynb)
2. ML Fundamentals [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/02_ml_fundamentals.ipynb)
3. Decision Trees [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/03_decision_trees.ipynb)
4. Neural Networks [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/04_neural_nets.ipynb)
5. Modern AI: LLMs, RAG & Agents [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/05_agentic_ai.ipynb)

### 🔌 Optional: an API key for Notebook 5

Notebook 5 has a handful of cells (marked 🔌) that call a **real** language model so you can
compare it with the toy model built in the notebook. They are entirely optional — without a
key those cells print a short note and everything else runs as normal.

```bash
cp .env.example .env     # then paste your key into .env
```

`.env` is listed in `.gitignore`, so your key never reaches the repository. On Colab, add
`OPENROUTER_API_KEY` under 🔑 **Secrets** in the left sidebar instead. Free keys:
[openrouter.ai/keys](https://openrouter.ai/keys).

> ⚠️ Never paste an API key into a notebook cell — notebook outputs get committed too.

To run notebooks locally (e.g. via Anaconda, Docker, or virtual environments), follow the instructions in:
📄 [INSTALLATION.md](INSTALLATION.md)


## 📘 What’s Inside?

These notebooks are designed to:

- Introduce essential machine learning concepts using real code
- Support hands-on sessions in our summer school programs
- Encourage experimentation and interdisciplinary exploration

They are designed from scratch but build on best practices and widely used teaching patterns.
We recommend the book below as a **complementary reference** for deeper dives and additional examples:

**Aurélien Géron – _Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_ (3rd ed., O'Reilly 2022)**
📚 GitHub: [ageron/handson-ml3](https://github.com/ageron/handson-ml3)


## 📂 Usage Guide

- 📖 Use [Google Colab](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/) for a zero-setup experience
- 💻 Or clone the repository and run notebooks locally
- ⚙️ Setup instructions for different platforms are available in [INSTALLATION.md](INSTALLATION.md)


## ⚖️ License

This repository is licensed under the **MIT License**.
See [LICENSE](LICENSE) for full details.

---

These materials are developed as part of the
**[Bridging AI & Society Summer Schools](https://github.com/BridgingAISocietySummerSchools)** initiative.
