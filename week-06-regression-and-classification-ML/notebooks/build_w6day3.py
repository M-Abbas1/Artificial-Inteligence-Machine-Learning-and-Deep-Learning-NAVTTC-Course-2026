import json, copy
import matplotlib; matplotlib.use("Agg")

def md(s): return {"k":"md","s":s}
def code(s): return {"k":"code","s":s}

CELLS = [

md("""# Week 6 · Day 3 — Logistic Regression: Predicting a Category

For two weeks we've predicted **numbers** — prices, marks, sales. Today the course turns a corner: we predict a **category**. Will a student *pass or fail*? Is an email *spam or not*? Is a tumor *benign or malignant*?

The model is **logistic regression**, and here's the good news up front: it's yesterday's straight line, gently transformed into a **probability**. Same spirit as everything so far — reuse what we know, add one clever piece.

**The plan:**
1. See the problem — why a straight line fails for yes/no.
2. The sigmoid — the function that squeezes any number into a probability.
3. Build logistic regression **from scratch** in NumPy.
4. Do it the real way with **scikit-learn**, and evaluate it."""),

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt"""),

# ============ 1. THE PROBLEM ============
md("""---
## 1. The problem: a yes/no answer

Our data: exam results. For each student we know their **study hours** and whether they **passed** (1) or **failed** (0). We want to predict pass/fail from study hours."""),

code("""df = pd.read_csv("exam_results.csv")
print("shape:", df.shape)
print("pass rate:", round(df["passed"].mean(), 2))
df.head()"""),

code("""# Look at it: every point is at height 0 (fail) or 1 (pass) — nothing in between
plt.scatter(df["study_hours"], df["passed"], alpha=0.5, color="purple")
plt.xlabel("study hours"); plt.ylabel("passed?  (0 = fail, 1 = pass)")
plt.title("The answer is a category, not a number")
plt.yticks([0, 1]); plt.grid(True, alpha=0.3)
plt.show()"""),

md("""Notice the shape: students who studied little cluster at the bottom (fail), students who studied a lot cluster at the top (pass), and there's a messy middle. The outputs are **only ever 0 or 1**.

Let's try fitting yesterday's straight line to this and see why it doesn't work."""),

code("""from sklearn.linear_model import LinearRegression

X = df[["study_hours"]].values
y = df["passed"].values
line = LinearRegression().fit(X, y)

xr = np.linspace(0, 12, 100)
plt.scatter(df["study_hours"], df["passed"], alpha=0.5, color="purple", label="data")
plt.plot(xr, line.predict(xr.reshape(-1, 1)), color="red", linewidth=2, label="straight line")
plt.axhline(0, color="gray", linestyle=":", alpha=0.7)
plt.axhline(1, color="gray", linestyle=":", alpha=0.7)
plt.xlabel("study hours"); plt.ylabel("passed?")
plt.title("A straight line is wrong for this job")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()"""),

md("""The line is **absurd here**:
- It goes **above 1** (for many study hours) and **below 0** (for very few) — but you can't be "130% likely to pass" or "−20% likely."
- It gives a smooth number, but we need a clean **yes/no**, ideally with a sense of *how confident* we are.

What we really want is a curve that:
- stays **between 0 and 1** (so we can read it as a probability),
- is **low** for few study hours, **high** for many, with a smooth S-shaped transition in between.

That curve exists. It's called the **sigmoid**."""),

# ============ 2. THE SIGMOID ============
md("""---
## 2. The sigmoid: squeezing any number into a probability

The **sigmoid** function takes *any* number — from minus infinity to plus infinity — and squeezes it into the range **0 to 1**. Its formula:

$$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$

It's one line of NumPy. Let's build it and look at it."""),

code("""def sigmoid(z):
    return 1 / (1 + np.exp(-z))"""),

code("""# Plot the sigmoid over a range of inputs
z = np.linspace(-10, 10, 200)
plt.plot(z, sigmoid(z), color="blue", linewidth=2)
plt.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
plt.axvline(0, color="gray", linestyle=":", alpha=0.7)
plt.xlabel("z (any number)"); plt.ylabel("sigmoid(z)  →  between 0 and 1")
plt.title("The sigmoid: the famous S-curve")
plt.grid(True, alpha=0.3)
plt.show()"""),

md("""Look at what it does:
- Feed it a **big positive** number → output near **1**.
- Feed it a **big negative** number → output near **0**.
- Feed it **0** → output exactly **0.5** (the midpoint).
- It never goes below 0 or above 1, and it transitions smoothly.

Let's confirm with a few values."""),

code("""for z_val in [-8, -2, 0, 2, 8]:
    print(f"sigmoid({z_val:3d}) = {sigmoid(z_val):.4f}")"""),

md("""This is exactly the tool we needed. Now the idea of **logistic regression** is simple:

> Take the straight line from before, `z = w·x + b`, and pass it through the sigmoid.

$$ p = \\sigma(w \\cdot x + b) $$

The line produces any number; the sigmoid squeezes it into a probability. That's the whole model — *"a straight line, squished into a probability."*"""),

code("""def predict_proba(x, w, b):
    z = w * x + b          # the straight line (any number)
    return sigmoid(z)      # squeezed into a probability 0-1"""),

md("""Let's see what this looks like for a hand-picked `w` and `b`, laid over the data. Even before training, you can see the S-curve is the *right shape* for this problem."""),

code("""w_guess, b_guess = 1.0, -5.0   # a hand-picked guess
xr = np.linspace(0, 12, 100)

plt.scatter(df["study_hours"], df["passed"], alpha=0.5, color="purple", label="data")
plt.plot(xr, predict_proba(xr, w_guess, b_guess), color="green", linewidth=2,
         label="logistic curve (a guess)")
plt.xlabel("study hours"); plt.ylabel("probability of passing")
plt.title("The logistic curve fits this shape naturally")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()"""),

md("""Already far more sensible than the straight line — it hugs 0 on the left, climbs through the middle, and flattens toward 1 on the right, never leaving the valid 0–1 range. Now we need to *train* it to find the best `w` and `b`, instead of guessing."""),

# ============ 3. FROM SCRATCH ============
md("""---
## 3. Training logistic regression from scratch

The training follows the **same gradient-descent recipe** as linear regression: define a cost (how wrong the model is), then take small downhill steps to reduce it. Two of the three pieces are identical to before; only the cost function changes.

### The cost function
For classification we don't use squared error. Instead we use **log loss** (also called cross-entropy). You don't need its algebra memorized — understand it by its *behavior*:
- If the true answer is **pass (1)** and the model says **0.99**, the cost is tiny (confident and right — good).
- If the true answer is **pass (1)** and the model says **0.01**, the cost is **huge** (confident and wrong — heavily punished).

In short: **log loss punishes confident mistakes hard.** That's exactly what we want."""),

code("""def compute_cost(x, y, w, b):
    p = predict_proba(x, w, b)
    eps = 1e-9                       # tiny value to avoid log(0)
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))"""),

md("""### The gradients
Remarkably, the gradient formulas come out looking **identical to linear regression** — the error `(prediction − actual)` times the feature. The maths behind this is elegant, but the practical point is: *you already know this loop.*"""),

code("""def compute_gradients(x, y, w, b):
    p = predict_proba(x, w, b)
    error = p - y                    # same "prediction minus actual" as before
    grad_w = np.mean(error * x)
    grad_b = np.mean(error)
    return grad_w, grad_b"""),

md("""### The training loop
Exactly the gradient-descent loop from Week 5 — start with a guess, step downhill, repeat."""),

code("""x = df["study_hours"].values
y = df["passed"].values

w, b = 0.0, 0.0          # start with a flat guess
learning_rate = 0.3
history = []

for epoch in range(1, 5001):
    grad_w, grad_b = compute_gradients(x, y, w, b)
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    history.append(compute_cost(x, y, w, b))

print(f"trained:  w = {w:.3f}   b = {b:.3f}")
print(f"final cost = {history[-1]:.4f}")"""),

code("""# The cost fell as it learned — same picture as always
plt.plot(history, color="purple")
plt.xlabel("epoch"); plt.ylabel("cost (log loss)")
plt.title("The cost falling during training")
plt.grid(True, alpha=0.3)
plt.show()"""),

md("""### The trained curve
Now let's draw the trained logistic curve over the data. This is the model that gradient descent found."""),

code("""xr = np.linspace(0, 12, 100)
plt.scatter(df["study_hours"], df["passed"], alpha=0.5, color="purple", label="data")
plt.plot(xr, predict_proba(xr, w, b), color="green", linewidth=2, label="trained model")
plt.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="decision threshold (0.5)")
plt.xlabel("study hours"); plt.ylabel("probability of passing")
plt.title("The trained logistic regression model")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()"""),

md("""### The decision boundary
The green curve gives a **probability**. To make an actual **decision**, we pick a threshold — usually **0.5**:
- probability ≥ 0.5 → predict **pass**
- probability < 0.5 → predict **fail**

The study-hours value where the curve crosses 0.5 is the **decision boundary** — the dividing line between our two predictions. It happens where `w·x + b = 0`, i.e. `x = -b/w`."""),

code("""boundary = -b / w
print(f"decision boundary: {boundary:.2f} study hours")
print(f"  study more than {boundary:.1f} hours → model predicts PASS")
print(f"  study less than {boundary:.1f} hours → model predicts FAIL")"""),

md("""### Using the model
Let's predict for individual students — both the probability and the yes/no decision."""),

code("""for hours in [2, 4, 5, 8]:
    p = predict_proba(hours, w, b)
    decision = "PASS" if p >= 0.5 else "FAIL"
    print(f"studies {hours:2d} hours → {p:.1%} likely to pass → predict {decision}")"""),

md("""There it is — a working classifier, built from scratch. It outputs a *probability* ("73% likely to pass") that we turn into a *decision*. Now, as always, let's see how the professionals do it."""),

# ============ 4. SCIKIT-LEARN ============
md("""---
## 4. The real way: scikit-learn

Everything we just built by hand — the sigmoid, the log-loss cost, the gradient-descent loop — is inside scikit-learn's `LogisticRegression`, ready in a few lines. And because you built it yourself, it's no longer a mystery."""),

code("""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix"""),

md("""### Train/test split (the golden rule)
As always, we hold back part of the data to test on — a model must be judged on data it hasn't seen."""),

code("""X = df[["study_hours"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print(f"training students: {len(X_train)},  testing students: {len(X_test)}")"""),

md("""### Train — three lines
The same four-step pattern as every scikit-learn model: create, fit, predict, score."""),

code("""model = LogisticRegression()
model.fit(X_train, y_train)
print("model trained.")
print(f"scikit-learn found:  w = {model.coef_[0][0]:.3f}   b = {model.intercept_[0]:.3f}")
print(f"decision boundary: {-model.intercept_[0]/model.coef_[0][0]:.2f} study hours")"""),

md("""Compare that boundary to the one our from-scratch model found — they land in the same place. Different code, same answer. That agreement is the moment the library earns your trust."""),

# ============ 5. EVALUATION ============
md("""---
## 5. Evaluating the classifier

For regression we used R² and RMSE. For **classification** the first tools are **accuracy** and the **confusion matrix**."""),

code("""predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"accuracy = {accuracy:.1%}  (fraction of test students predicted correctly)")"""),

md("""### The confusion matrix
Accuracy is one number; the **confusion matrix** shows the *full* picture — exactly which kinds of mistakes the model made. It's a 2×2 table:

|  | predicted FAIL | predicted PASS |
|---|---|---|
| **actually FAIL** | ✓ correct | ✗ false alarm |
| **actually PASS** | ✗ missed | ✓ correct |

The diagonal (top-left, bottom-right) is correct predictions; off-diagonal is mistakes."""),

code("""cm = confusion_matrix(y_test, predictions)
print("confusion matrix:")
print(cm)

# a labeled heatmap makes it easier to read (Week 5 Seaborn skill)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["pred FAIL", "pred PASS"],
            yticklabels=["true FAIL", "true PASS"])
plt.title("Confusion matrix")
plt.show()"""),

md("""Read the grid: the two diagonal cells are students the model got right; the off-diagonal cells are the two kinds of error it made. For pass/fail, a *false alarm* (predicting pass when they failed) and a *miss* (predicting fail when they passed) are different mistakes — and which one matters more depends on the situation. We'll dig into precision, recall, and F1 tomorrow, when we have several classes to juggle."""),

code("""# The full scikit-learn model, drawn over ALL the data
xr = np.linspace(0, 12, 100)
plt.scatter(df["study_hours"], df["passed"], alpha=0.4, color="purple", label="data")
plt.plot(xr, model.predict_proba(xr.reshape(-1, 1))[:, 1], color="green",
         linewidth=2, label="logistic model")
plt.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="threshold")
plt.xlabel("study hours"); plt.ylabel("probability of passing")
plt.title("scikit-learn logistic regression")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()"""),

# ============ SUMMARY ============
md("""---
## Summary

- **Classification** predicts a category (pass/fail), where **regression** predicts a number. A straight line fails for yes/no because it isn't bounded between 0 and 1.
- The **sigmoid** `σ(z) = 1/(1+e⁻ᶻ)` squeezes any number into a probability between 0 and 1.
- **Logistic regression** = a straight line passed through the sigmoid: `p = σ(w·x + b)`. It outputs a *probability*.
- Training uses the **same gradient descent** as linear regression, but with a **log-loss** cost that punishes confident wrong answers hard.
- A **decision boundary** (where probability = 0.5) turns the probability into a yes/no decision.
- **scikit-learn's `LogisticRegression`** does all of this in a few lines — and finds the same boundary we found by hand.
- Evaluate a classifier with **accuracy** and the **confusion matrix**, which reveals *which* mistakes it makes.

Next time: **multiclass classification** — what happens when there are more than two categories (three flower species, ten digits) — and the fuller evaluation toolkit of precision, recall, and F1."""),
]

BASE={"nbformat":4,"nbformat_minor":5,
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
             "language_info":{"name":"python","version":"3.11"}},"cells":[]}
def mk(c,i):
    if c["k"]=="md": return {"cell_type":"markdown","id":f"c{i:03d}","metadata":{},"source":c["s"]}
    return {"cell_type":"code","id":f"c{i:03d}","metadata":{},"execution_count":None,"outputs":[],"source":c["s"]}
nb=copy.deepcopy(BASE); nb["cells"]=[mk(c,i) for i,c in enumerate(CELLS)]
json.dump(nb,open("w6day3-logistic-regression.ipynb","w"),indent=1)
print("built:",len(nb["cells"]),"cells")
