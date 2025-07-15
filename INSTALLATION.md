
# INSTALLATION.md

## 🛠️ How to Set Up the Hands-On ML Notebooks Locally

This guide walks you through installing everything you need to run the notebooks in this repository. We recommend using **Anaconda**, a free Python distribution that includes package management and environment setup tools.

If you prefer not to install anything, you can use Google Colab instead:
👉 [Open in Colab](https://colab.research.google.com/github/BridgingAISocietySummerSchools/Hands-On-Notebooks/)

---

## 1. 📦 Installing Anaconda

Anaconda is available for Windows, macOS, and Linux from:
👉 [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)

### Choose the installer for your operating system:
- Windows/macOS: use the **graphical installer**
- Linux: use the **command-line installer**

> We strongly recommend installing Anaconda **for your user only** (not system-wide).
> On Linux/macOS, this typically installs to `~/anaconda3`.

Official guides:
- [Installing on Windows](https://docs.anaconda.com/anaconda/install/windows/)
- [Installing on macOS](https://docs.anaconda.com/anaconda/install/mac-os/)
- [Installing on Linux](https://docs.anaconda.com/anaconda/install/linux/)

Once installed:
- On Windows/macOS, verify installation by launching **Anaconda Navigator**
- On Linux/macOS CLI, follow the prompt to run `conda init`, then restart your shell
  Test with:
  ```bash
  conda list
  python  # You should see "Anaconda" in the version banner
  quit()
  ```

---

## 2. 🧪 Setting Up the ML Environment

Once Anaconda is installed, you can set up the environment in two ways:

### 2a. Anaconda Navigator (GUI)

**Recommended for beginners on Windows/macOS**

1. Launch **Anaconda Navigator**
2. Go to **Environments → Import**
3. Select the file `ml-environment.yml` from this repository
4. Give the environment a name (e.g., `ml-workshop`) and click Import
5. Once created, activate the environment and launch **Jupyter Notebook** from the GUI

---

### 2b. Conda via Command Line (Linux/macOS, advanced users)

```bash
# Clone or download this repository
git clone https://github.com/BridgingAISocietySummerSchools/Hands-On-Notebooks.git
cd Hands-On-Notebooks

# Create environment
conda env create -f ml-environment.yml

# Activate it
conda activate ml-workshop

# Launch notebooks
jupyter notebook
```

---

## 3. 🐳 Docker (Optional for Experts)

You can run a full environment in Docker without installing anything else.

Build the image:
```bash
docker build -t ml-workshop-image .
```

Then launch the notebook server:
```bash
docker run --rm -u $(id -u):$(id -g) -p 8888:8888 -v $PWD:/data ml-workshop-image
```

Access Jupyter via [http://localhost:8888/](http://localhost:8888/)
Copy the token from the terminal output when launching the container.

> Note: This binds the current directory into the container, so changes you make to notebooks will be saved.

---

## 4. 🧠 Expert Setup (No Anaconda)

If you already use virtual environments or your own Python setup, you can install required packages directly using `pip`.

### Requirements
Install the packages listed in `ml-environment.yml`, in particular:

- `scikit-learn` ≥ 1.2
- `tensorflow` ≥ 2.15

Then launch Jupyter:
```bash
jupyter notebook
```

Start with `01_test_notebook.ipynb` to verify that everything works.

---

_Last updated: July 2025_
