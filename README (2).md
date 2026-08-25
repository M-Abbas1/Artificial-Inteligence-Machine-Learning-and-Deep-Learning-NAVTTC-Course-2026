# Artificial Intelligence, Machine Learning & Deep Learning

A hands-on course taking you from Python basics to training neural networks, fine-tuning Transformers, and deploying a working AI app — built on the NAVTTC *Skills for All* AI/ML/DL curriculum and modernized for 2026.

Everything you build lives in this repository, organized week by week. By the end you'll have a public repo full of real, runnable projects — a portfolio, not just notes.

---

## What you'll learn

The course is **20% theory, 80% hands-on**. Concepts are opened with a short explanation and then immediately closed with running code — the last thing you see before every break is code, not a slide.

By the end you will be able to:

- Write clean Python and work fluently with NumPy, Pandas, and visualization libraries
- Understand the statistics and probability that underpin machine learning
- Build, train, and evaluate classical ML models with scikit-learn
- Build neural networks **from scratch** and in **PyTorch** (our main framework), with a look at TensorFlow/Keras so you can read either
- Work with CNNs, RNNs/LSTMs, word embeddings, and the Transformer architecture
- Use Hugging Face pretrained models and fine-tune one on your own data
- Apply Generative AI, prompt engineering, and RAG basics
- Use Azure AI services
- **Ship a model as a live web app** with a public demo link

Deep learning is taught **PyTorch-first**. You build a network by hand (neurons, backprop, the training loop) before any framework, so every line of PyTorch later maps onto something you already understand.

---

## Course roadmap

| Week | Folder | Focus |
|------|--------|-------|
| **1** | `week-01-foundations-linux-python` | AI foundations, Linux shell, Python basics, Git & GitHub from day one |
| **2** | `week-02-data-structures-control-flow-functions` | Data structures, control flow, functions |
| **3** | `week-03-file-exception-handling-oops` | File & exception handling, OOP |
| **4** | `week-04-libraries-descriptive-statistics` | NumPy, Pandas, Seaborn & descriptive statistics |
| **5** | `week-05-visualization-and-first-models-ML` | Visualization & first ML models |
| **6** | `week-06-regression-and-classification-ML` | Regression & classification |
| **7** | `week-07-completing-classical-ML` | Completing classical ML (trees, ensembles, SVM) |
| **8** | `week-08-deep-learning-1-neural-networks` | Deep Learning I — neural networks from scratch → PyTorch |
| **9** | `week-09-deep-learning-2` | Deep Learning II — CNNs, sequence models & more |
| **10–12** | *upcoming* | Fine-tuning Transformers · Generative AI & Azure AI · deployment & final project |

---

## Repository structure

```
Artificial-Inteligence-Machine-Learning-and-Deep-Learning/
├── .vscode/                    editor settings
├── .gitignore
├── 00-course-admin/            course plan & admin docs
├── week-01-foundations-linux-python/
│   ├── class-live-code/        code written live in class
│   ├── exercises/              practice tasks
│   ├── notebooks/              lesson notebooks
│   └── slides/                 slide handouts
├── week-02-data-structures-control-flow-functions/
│   └── (same four folders)
├── ...
└── week-09-deep-learning-2/
    ├── class-live-code/
    ├── datasets/               data used this week (large files git-ignored)
    ├── exercises/
    ├── notebooks/
    └── slides/
```

Each week follows the same pattern: **`notebooks/`** (the lesson material), **`class-live-code/`** (what we type together in class), **`exercises/`** (your practice), and **`slides/`**. Weeks that use their own data add a **`datasets/`** folder.

---

## Getting started

### 1. Prerequisites

- **Anaconda** (bundles Python + conda + the scientific libraries) — [download](https://www.anaconda.com/download)
- **Git** — [download](https://git-scm.com/downloads)
- A **GitHub account**
- An editor: **VS Code** or **PyCharm**

> New to virtual environments and conda? See the **Setting Up Your Python Environment** handout — it walks through every step and the common install errors.

### 2. Get the code

```bash
git clone https://github.com/<your-username>/Artificial-Inteligence-Machine-Learning-and-Deep-Learning.git
cd Artificial-Inteligence-Machine-Learning-and-Deep-Learning
```

### 3. Create the environment

```bash
conda create -n ai-course python=3.11
conda activate ai-course
conda install numpy pandas matplotlib seaborn scikit-learn jupyter pillow
pip install torch
```

Install extras as later weeks need them (e.g. `pip install transformers streamlit`).

### 4. Check it worked

```bash
python -c "import torch; print('PyTorch', torch.__version__)"
```

If that prints a version number, you're ready.

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open the current week's `notebooks/` folder and start there.

---

## Datasets

Some weeks load image or text datasets that live in folders on disk (read with Python's `os`) rather than a built-in loader — for example an image set arranged one folder per class:

```
DatasetName/
  train/
    class_a/   img1.jpg, img2.jpg, ...
    class_b/   ...
  test/
    class_a/   ...
    class_b/   ...
```

Point the notebook's `DATA_DIR` at the folder that contains the class subfolders. In a firewalled lab, datasets are provided on a USB stick — copy the dataset folder into that week's `datasets/` directory (or next to the notebook) before running. **Large datasets are not committed** to this repo — they're covered by `.gitignore`.

---

## Weekly workflow

1. **Pull** the latest material at the start of each week:

   ```bash
   git pull
   ```

2. **Work** through that week's `notebooks/` and `exercises/`. Many notebooks are scaffolded with `TODO`s — you fill in the model, the training loop, and the experiments yourself.

3. **Push** your completed work at the end of the week:

   ```bash
   git add week-0X-.../
   git commit -m "Week X: exercises + notes"
   git push
   ```

Pushing your work each week *is* the real Git workflow — it costs no extra time and builds a portfolio you can show employers.

---

## A note on effort and honesty

Some datasets in this course are genuinely hard (facial-emotion recognition, for instance, tops out well below 100% for a simple network). That's intentional. You'll see honest results, understand *why* a plain network hits a ceiling, and learn what the next tool — a CNN, a Transformer — fixes. Chasing understanding beats chasing accuracy.

---

## Tools & libraries

**Python · Jupyter · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn · PyTorch** (primary) · **TensorFlow/Keras** (for reading both) · **Hugging Face Transformers · Streamlit / Gradio · Azure AI · Git & GitHub**

---

*Based on the NAVTTC "Skills for All" AI/ML/DL curriculum, revised into a code-first redesign.*
