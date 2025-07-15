# Hands-On-Notebooks 📓🧑‍💻

> ⚠️ _This repository is still being updated. Expect occasional changes in structure, content, and setup instructions._
> _Last updated: July 2025_

![Build Status](https://github.com/knutzk/ml-workshop/actions/workflows/build_run_test.yml/badge.svg)

A collection of Jupyter notebooks to teach you the **basics of machine learning**.
They provide out-of-the-box code examples to explore and understand key ML algorithms.

---

## 📘 About This Repository

Almost all code examples are taken from or inspired by:

> [3] A. Géron, _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_, 3rd edition, O'Reilly 2022, ISBN: 978-1098125974
> [GitHub repository](https://github.com/ageron/handson-ml3)

We recommend studying the book alongside these notebooks.
To acknowledge Géron’s work, this repository uses the same **Apache 2.0 License** — please follow its terms when using or distributing this content.

---

## 🚀 Run on Google Colab

Want to run the notebooks without installing anything locally?
Use [Google Colab](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/) — all you need is a Google account.

📎 [Open with Google Colab](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/)

---

## 🧪 Viewing the Notebooks

You can explore notebook contents in your browser via:

- [Google Colab](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/)
- [Jupyter nbviewer](https://nbviewer.jupyter.org/github/knutzk/ml-workshop/)
- Or simply open any notebook directly on GitHub

For pre-executed notebooks, check the `md_output/` directory.

---

## ⚙️ Installing Python

We recommend using **Anaconda** or **virtual environments**.

Detailed setup instructions are in [INSTALLATION.md](INSTALLATION.md), including:

- Anaconda Navigator setup (Windows/macOS)
- Conda CLI setup (Linux/macOS)
- Docker-based installation
- Expert setup without Anaconda

---

### 📥 1. Obtain a Copy of This Repository

Via GitHub:

- Click the green **Code** button and select “Download ZIP”
- Unpack the archive on your system

Via `git`:

```bash
git clone https://github.com/knutzk/ml-workshop.git
```

### 🧭 2a) Setup with Anaconda Navigator (Windows/macOS)

1. Open Anaconda Navigator
2. Go to Environments → Import
3. Select ml-environment.yml from this repo
4. Name the environment (e.g., ml-workshop)
5. Import and wait for installation
6. Activate → Open with Jupyter Notebook

### 💻 2b) Setup with Conda (Linux/macOS)

```bash
# Activate base conda if needed
source <path-to-conda>/bin/activate
conda init

# Create and activate the environment
cd <path-to-repo>
conda env create -f ml-environment.yml
conda activate ml
jupyter notebook
```

### 🧪 3) Run the First Notebook

Open `01_test_notebook.ipynb` and follow the instructions to verify your installation.

To stop Jupyter (if running via terminal), use CTRL+C.
To deactivate the environment:

```bash
conda deactivate
```

### 🐳 Docker Setup (Optional)

To build the container:

```bash
cd <path-to-repo>
docker build -t ml-workshop-image .
```

To run it:

```bash
docker run --rm -u $(id -u):$(id -g) -p 8888:8888 -v $PWD:/data ml-workshop-image
```

Visit http://localhost:8888
Use the token printed in the terminal.

### 🧠 Expert Setup (Without Anaconda)

You may use your own virtual environments, `venv`, or tools like `pyenv`.

Install at minimum:
- scikit-learn 1.2.x
- tensorflow 2.15.x

Then launch jupyter notebook in the repo directory and open the files directly.
