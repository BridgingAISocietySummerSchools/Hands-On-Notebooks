# INSTALLATION.md

## 🚀 Quick Start: Run on Google Colab

If you prefer not to install anything locally, run the notebooks directly in your browser using **Google Colab**:

👉 [Open in Colab](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/)

---

## 🛠️ Option 1: Anaconda (Recommended for Beginners)

We recommend installing [**Anaconda**](https://www.anaconda.com/distribution/) to manage packages and environments easily. It works on Windows, macOS, and Linux.

### 📥 Step 1: Install Anaconda

Choose the installer for your operating system:
- **Windows/macOS:** use the graphical installer
- **Linux:** use the command-line installer

> ✅ Tip: Install Anaconda *for your user only* (not system-wide).
> On Linux/macOS, this typically installs to `~/anaconda3`.

Useful guides:
- [Installing on Windows](https://docs.anaconda.com/anaconda/install/windows/)
- [Installing on macOS](https://docs.anaconda.com/anaconda/install/mac-os/)
- [Installing on Linux](https://docs.anaconda.com/anaconda/install/linux/)

### 🧪 Step 2: Set Up the Environment

Use the following commands to create and activate an environment with the required packages:

```bash
# Clone the repository
git clone https://github.com/BridgingAISocietySummerSchools/Hands-On-Notebooks.git
cd Hands-On-Notebooks

# Create and activate the environment
conda create -n ml-workshop python=3.11
conda activate ml-workshop

# Install the required packages
pip install -r requirements.txt

# Launch the notebooks
jupyter notebook
```

---

## 🐍 Option 2: pip + virtualenv (Lightweight Alternative)

If you prefer to avoid Anaconda, you can use Python's built-in tools. This approach is ideal if you're already comfortable managing environments.

```bash
# Clone the repository
git clone https://github.com/BridgingAISocietySummerSchools/Hands-On-Notebooks.git
cd Hands-On-Notebooks

# Create a virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate       # macOS/Linux
# .\venv\Scripts\activate       # Windows

# Install the required packages
pip install -r requirements.txt

# Launch the notebooks
jupyter notebook
```

### 💡 Using pyenv + pyenv-virtualenv (Advanced Users)

If you're using [`pyenv`](https://github.com/pyenv/pyenv) with [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv), you can also create and manage your environment this way:

```bash
# Select or install a specific Python version
pyenv install 3.11.13  # if not already installed
pyenv virtualenv 3.11.13 ml-workshop
pyenv activate ml-workshop

# Move into the project directory
cd Hands-On-Notebooks

# Install requirements
pip install -r requirements.txt

# Launch notebooks
jupyter notebook
```

This gives you per-project Python version control and integrates well with `.python-version` files.

---

## 🐳 Option 3: Docker (Advanced/Isolated Setup)

Use Docker to run everything in a containerized environment:

### Step 1: Build the Docker Image

```bash
docker build -t ml-workshop-image .
```

### Step 2: Run the Notebook Server

```bash
docker run --rm -u $(id -u):$(id -g) -p 8888:8888 -v $PWD:/data ml-workshop-image
```

Then open your browser at: [http://localhost:8888/](http://localhost:8888/)
Copy the token from the terminal output when prompted.

> 💾 The current directory is mounted into the container, so your work is saved outside the container.

---

## 🔑 Optional: API Key for Notebook 5

`05_agentic_ai.ipynb` has a few cells (marked 🔌) that call a real language model. They are
optional — without a key those cells print a note and skip, and everything else runs normally.

```bash
cp .env.example .env      # then paste your key into .env
```

`.env` is listed in `.gitignore`, so the key stays on your machine. On **Colab** there is no
`.env` file: add `OPENROUTER_API_KEY` under the 🔑 **Secrets** icon in the left sidebar instead.
Free keys are available at [openrouter.ai/keys](https://openrouter.ai/keys).

> ⚠️ Never type an API key directly into a notebook cell. Notebook outputs are committed too.

---

## ✅ Verifying the Setup

After installing, open and run:

```text
01_test_notebook.ipynb
```

If the notebook runs without error, your setup is complete.

---

## 📊 Setup Options Summary

| Method            | Recommended For         | Setup Style             |
|------------------|-------------------------|-------------------------|
| Google Colab     | Everyone (quick start)  | No installation         |
| Anaconda         | Beginners (Win/macOS)   | Full-featured GUI/CLI   |
| pip + virtualenv | Python-savvy users      | Lightweight, flexible   |
| pyenv            | Advanced CLI users      | Version-controlled envs |
| Docker           | Experts, reproducibility| Isolated containers     |

---

_Last updated: July 2025_
