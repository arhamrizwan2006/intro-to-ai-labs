# Lab 01 Report — Environment Setup: Anaconda, Jupyter Notebook and Git

**Course:** Introduction to Artificial Intelligence (AI-101L)
**Student:** Arham Rizwan

## 1. Installation evidence

- Anaconda installed from anaconda.com/download with default options.
- `ai-lab` conda environment created (`python=3.11`) and registered as a Jupyter kernel
  (`Python (ai-lab)`), shared across all labs in the course.
- Git installed from git-scm.com/downloads and identity configured (`user.name`, `user.email`).
- *(Screenshot of the kernel selected inside Jupyter — required for Exercise 1 — kept alongside
  this report on my own machine; not embedded here to keep the repository lightweight.)*

## 2. Dataset inspection (Task 1.3)

Dataset: Titanic passenger list, loaded directly from
`https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv`

**Shape:** 891 rows × 15 columns

**Columns with missing values (Exercise 2):**

| Column   | Missing | Percent |
|----------|---------|---------|
| deck     | 688     | 77.2%   |
| age      | 177     | 19.9%   |
| embarked | 2       | 0.2%    |

**Coefficient of variation, largest first (Exercise 3):**

| Column   | std / mean |
|----------|------------|
| parch    | 2.112      |
| sibsp    | 2.108      |
| fare     | 1.543      |
| survived | 1.268      |
| age      | 0.489      |
| pclass   | 0.362      |

**Answer:** `parch` has the largest coefficient of variation. Models that work directly on raw
numeric magnitude (k-NN, SVMs, gradient-descent-based regressors) will let high-variance columns
like `parch` and `fare` dominate the distance/gradient calculation purely because of their scale,
while a lower-variance but still useful column like `pclass` gets effectively drowned out — this
is exactly why features are normalized before training on unscaled data.

## 3. Git graph (Exercise 4)

Real output of `git log --oneline --graph --all` from my `ai-lab-portfolio` repository:

```
* 1191ae9 Lab 01: add report with real git graph
* d53f722 Chore: ignore __pycache__, drop scratch exercise log
*   418498f Merge branch 'lab01/homework' into main
|\  
| * bd21cf7 Lab 01 Homework: inspect() function, run on Titanic and Iris
|/  
*   bf04f7b Merge branch 'lab01/exercises' into main
|\  
| * 208eb10 Lab 01 Exercise 4: practice branch-commit-merge cycle
| * 3a8a8b9 Lab 01 Exercise 3: coefficient of variation analysis
| * 1d7347e Lab 01 Exercise 2: identify Titanic columns with missing values
|/  
* 8696dcb Lab 01: environment setup and first notebook
```

(Graph taken just before this report was committed.)

Three commits were made on `lab01/exercises`, then merged into `main`; a second branch,
`lab01/homework`, was used for the home assignment below and also merged in.

## 4. Home assignment (`inspect.py`)

`lab01/inspect.py` defines `inspect(path_or_url)`, which loads any CSV and prints one compact
report: shape, dtypes, per-column missing counts/percentages, and the numeric summary.

Run against two different public datasets on the `lab01/homework` branch:

1. **Titanic** (891 × 15) — 3 columns with missing data (see table above)
2. **Iris** (150 × 5) — no missing values

Full console output of both runs is saved in `lab01/inspect_output.txt`.
