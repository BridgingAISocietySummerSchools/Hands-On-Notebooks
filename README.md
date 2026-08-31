# Hands-On Notebooks 📓🧑‍💻

[![Python Version](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/)
[![Difficulty](https://img.shields.io/badge/difficulty-beginner-success)](#)
[![Duration](https://img.shields.io/badge/duration-3%20hours-brightgreen)](#)
![Docker + Notebook Test](https://github.com/BridgingAISocietySummerSchools/Hands-On-Notebooks/actions/workflows/build_run_test.yml/badge.svg)


A curated collection of **Jupyter notebooks** to explore and teach the fundamentals of **machine learning**. These notebooks are practical, beginner-friendly, and support our interdisciplinary summer school curriculum.


## 🚀 Quick Start

You can run all notebooks in your browser via **Google Colab** — no installation required:
👉 [Open in Google Colab](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/)

0. Test Notebook [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/00_test_notebook.ipynb)
1. Classification & Evaluation [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/01_classification.ipynb)
2. Regression & Gradient Descent [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/02_regression.ipynb)
3. Decision Trees [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/03_decision_trees.ipynb)
4. Neural Networks [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/04_neural_nets.ipynb)
5. Modern AI in Practice — RAG in 90 minutes [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/05_agentic_ai.ipynb)

Notebook 5 comes in two versions:

| Version | Duration | For |
|---------|----------|-----|
| [`05_agentic_ai.ipynb`](05_agentic_ai.ipynb) | ~90 min | **The taught session.** Application-focused, RAG at the centre, five short hands-on tasks. Calls a real model throughout. |
| [`self_learning/05_agentic_ai.ipynb`](self_learning/05_agentic_ai.ipynb) [![Run in Colab](https://img.shields.io/badge/run%20in-colab-yellow)](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/blob/main/self_learning/05_agentic_ai.ipynb) | ~3 h | **Self-study deep dive.** Builds a language model from scratch, four failure modes, RAG, agents, memory, multi-agent systems, prompt injection. Runs entirely offline. |

### 🔑 An API key for Notebook 5

Notebook 5 talks to a **real** language model through [OpenRouter](https://openrouter.ai/).

- **In the taught 90-minute session**, the instructor hands out a key. The setup cell asks for
  it in a **hidden input box** (`getpass`), so it stays in memory and never lands in the file.
- **In the self-study version**, those cells are marked 🔌 and entirely optional — without a
  key they print a short note and everything else runs as normal.

To use your own key instead:

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
