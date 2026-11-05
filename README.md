# Introduction to AI - Labs (AI-101L)

Arham, BS Robotics & AI, University of Lahore. Instructor: Dr. Muhammad Awais.
Every notebook is executed with its real outputs saved. Each lab was developed on its own branch and merged into `main`.

| Lab | Topic | Folder | Headline result |
|---|---|---|---|
| 1 | Environment setup (Anaconda, Jupyter, Git) | `lab01/` | Working `ai-lab` kernel, screenshots, git history |
| 2 | Web scraping and EDA | `lab02/` | Scraped articles saved to CSV, EDA notebook |
| 3 | Feature extraction from text and images | `lab03/` | Bag-of-words and 4096-pixel image features |
| 4 | Linear regression (simple and multiple) | `lab04/` | Salary R2 0.90; house price R2 0.998 |
| 5 | Classification (diabetes, Titanic) | `lab05/` | Diabetes 72.7%; Titanic tuned RF 81.0% vs 61.5% baseline |
| 6 | Model deployment with FastAPI | `lab06/` | 11 tests pass, about 8.5 ms per prediction |
| 7 | CNN layers | - | Not included (no material provided) |
| 8 | MNIST digit recognition (CNN) | `lab08/` | 98.7% validation accuracy |

Per-lab details are in each folder's `report.md` where one exists. Lab 8 needs `train.csv` and `test.csv` (Kaggle Digit Recognizer) in `lab08/data/` to rerun; they are not committed because of size.
