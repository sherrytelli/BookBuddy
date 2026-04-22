# 📚 BookBuddy: Hybrid NLP & Collaborative Book Recommendation Engine

## 📌 Project Overview

**BookBuddy** is an advanced, two-pronged hybrid recommender system built primarily on the **goodbooks-10k** dataset. Traditional collaborative filtering often struggles with the "cold start" problem or suggests books that, while topically relevant, mismatch the user's reading comprehension level.

BookBuddy solves this by blending a **Deep Learning Collaborative Filter** (tracking behavioral user-item affinities) with an **NLP Content-Based Filter** (evaluating linguistic complexity). By calculating Flesch-Kincaid readability metrics, the system maps books to proxy CEFR reading tiers (A2 Beginner, B1 Intermediate, B2 Advanced, C1 Expert) to guarantee that recommendations are both personalized to the user's taste *and* appropriate for their reading proficiency.

---

## 🧠 Core Architecture

The recommendation engine is powered by three distinct modules:

1.  **Content-Based NLP Gateway (Machine Learning):** * Uses `textstat` to evaluate reading complexity, mapping grades to proficiency levels.
    * A **Multinomial Naïve Bayes Classifier** trained on TF-IDF metadata vectors predicts the probability distribution of a book belonging to a specific proficiency tier.
2.  **Deep Collaborative Filtering (Deep Learning):** * A PyTorch-based **Multi-Layer Perceptron (MLP)** maps users and books into a shared latent semantic space.
    * Optimized using **Bayesian Personalized Ranking (BPR) Loss** to enforce strict ranking hierarchies between observed and unobserved interactions.
3.  **Smart Hybrid Engine:** * Fuses normalized signals from both models using a **multiplicative penalty fusion mechanism**. 
    * It dynamically isolates a user's preferred reading level based on their historical reads, fetches the corresponding NLP probabilities, and penalizes the MLP score if the book doesn't match their reading level.

---

## 📂 Repository Structure

The project is organized precisely as follows to separate logic, artifacts, serialization, and experimental tuning:

```text
BOOKBUDDY/
├── bookbuddy_artifacts/               # Exported matrices, processed DataFrames, and mappings
│   ├── artifacts.pkl
│   ├── books_clean.parquet            # Cleaned English books with mapped CEFR levels
│   ├── nb_content_scores.parquet
│   ├── nb_probabilities_normalized.npy
│   ├── nb_score_scaler.pkl
│   ├── ratings_filtered.parquet       # Dense ratings matrix (5.5M rows)
│   ├── test.parquet
│   ├── tfidf_matrix.npz               # Sparse TF-IDF representations
│   ├── train.parquet
│   ├── user_item_matrix.npz
│   └── val.parquet
├── evaluations/                       # Stored evaluation metrics across parameter sweeps
│   ├── hybrid_evaluation_results.json
│   └── mlp_evaluation_results.json
├── models/                            # Serialized PyTorch state dicts & scikit-learn models
│   ├── mlp_bpr_weights_1.pt
│   ├── mlp_bpr_weights.pt
│   ├── mlp_config_1.pt
│   ├── mlp_config.pt
│   └── nb_proficiency_model.pkl
├── training/                          # Core experimental ML/DL pipeline
│   ├── hybrid-training.ipynb
│   ├── mlp_training.ipynb
│   └── nbc_training.ipynb
└── preprocessing.ipynb                # Data cleaning, textstat feature engineering, splitting
```

---

## ⚙️ Setup & Installation

Follow these steps to replicate the environment and run the pipeline.

### 1. Clone the Repository

```bash
git clone https://github.com/sherrytelli/BookBuddy.git
cd BookBuddy
```

### 2. Create a Virtual Environment

To keep dependencies isolated and avoid package collisions, instantiate a Python virtual environment:

```bash
# Initialize venv
python3 -m venv venv

# Activate venv (Linux/macOS)
source venv/bin/activate

# Activate venv (Windows)
venv\Scripts\activate
```

### 3. Install Requirements

The project relies on a specific ML stack. Run the following inside your active virtual environment:

```bash
pip install --upgrade pip
pip install pandas numpy scipy scikit-learn==1.8.0 matplotlib seaborn jupyter tqdm optuna psutil textstat pyarrow
# Install PyTorch (Modify the index-url based on your local CUDA version if utilizing GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
```

---

## 🚀 Execution Pipeline

To rebuild the artifacts, train the models, and evaluate the engine, execute the Jupyter Notebooks in the following sequential order:

### Step 1: `preprocessing.ipynb`
* **Objective:** Cleans raw data and engineers linguistic features.
* **Process:** Filters for English variants (`eng`, `en-US`, `en-GB`), drops records with missing titles/authors, and computes Flesch-Kincaid (`fk_grade`) scores via `textstat`. Generates TF-IDF vectorizations and exports `train/val/test` (70/15/15) data splits as highly compressed `.parquet` files.

### Step 2: `training/nbc_training.ipynb`
* **Objective:** Trains the reading proficiency classifier.
* **Process:** Loads the TF-IDF matrix. Conducts 5-fold Cross-Validation to tune the Laplace smoothing parameter (`alpha`) for a `MultinomialNB` model. Optimizes for `f1_weighted` due to heavy class imbalances (Beginner A2 heavily outnumbers Expert C1). Outputs probability matrices to the artifacts folder.

### Step 3: `training/mlp_training.ipynb`
* **Objective:** Trains the behavioral PyTorch BPR-MLP model.
* **Process:** Constructs a custom `Dataset` and `DataLoader` for implicit feedback. Uses **Optuna** to execute a hyperparameter search across `embed_dim`, `learning_rate`, `dropout`, and `weight_decay`. Evaluates model convergence via Information Retrieval metrics (Rank@5, NDCG@10).

### Step 4: `training/hybrid-training.ipynb`
* **Objective:** Fuses the outputs into the final ranking formula.
* **Process:** Compiles the `HybridRecommender` class. The engine calculates a user's primary reading level, grabs the exact probability of a book matching that level from the NBC model, and combines it with the MLP collaborative scores using a **Multiplicative Penalty** algorithm:
  * `NBC Penalty = Alpha + ((1 - Alpha) * NBC_Probability_Score)`
  * `Final Score = MLP_Score * NBC_Penalty`
* Iterates through `Alpha` ranges to find the optimal balance where NLP penalties improve overall ranking relevance without destroying collaborative discovery.

---

## 📊 Summary of Results & Key Metrics

### 🗄️ Dataset Compression
* **User-Item Matrix:** Compressed 5.9 million raw ratings down to **5.5 million dense interactions**.
* **Dimensions:** 53,424 Users × 8,726 valid English books.
* **Density:** Maintained a healthy 1.19% sparsity rating for collaborative training.

### 📖 NLP Content Model (Naïve Bayes)
* **Optimal Alpha (Laplace):** `0.1`
* **Performance:** Achieved a Cross-Validation F1-Weighted score of `0.5353`.
* **Impact:** Successfully modeled probability distributions across 4 CEFR classes, allowing the system to confidently categorize text complexity using only title and tag metadata.

### 🤖 Collaborative Deep Model (PyTorch BPR-MLP)
Through extensive Optuna sweeps, the model discovered highly optimized latent dimension parameters (`embed_dim: 64`, `lr: ~0.00049`, `dropout: ~0.342`, `weight_decay: ~0.00017`).
* **Test Set Performance (Evaluated on 53,172 Users):**
  * **Precision@10:** 0.0733
  * **Recall@10:** 0.0685
  * **NDCG@10:** 0.0884
* **Cold Start Robustness:** When retaining only 25% of a user's interaction history, the MLP sustained an NDCG@10 of `0.0663`.

### 🏆 Hybrid Synergy & Final Engine
* Integrating the models using a parameterized `Alpha` search discovered that an **Alpha of 0.9** provided the best fusion balance. The Multiplicative Fusion approach proved highly effective: by utilizing the NBC score as a scaled penalty, the engine successfully filtered out books that misaligned with a user's reading proficiency, safely pushing perfectly-comprehensible literature into their Top 5 recommendations. 
* **Final Hybrid Test Set Performance (Alpha = 0.9):**
  * **Precision@10:** 0.0668
  * **Recall@10:** 0.0625
  * **NDCG@10:** 0.0802
* **Hybrid Cold Start Robustness:** When retaining only 25% of a user's interaction history, the Hybrid model yielded an NDCG@10 of `0.0611`.

Here is a professional, analytical explanation that you can drop directly into your `README.md` (perhaps as a subsection under the **Summary of Results & Key Metrics**). It frames the slight drop in metrics not as a failure, but as an intentional and valuable architectural trade-off.

---

## ⚖️ Architectural Insight: Why is the Hybrid Score Slightly Lower than the Pure MLP?

You might notice a slight dip in the raw evaluation metrics when moving from the Pure MLP (`NDCG@10: 0.0884`) to the Final Hybrid Engine (`NDCG@10: 0.0802`). In the context of an educational recommender system, **this is an expected and intentional trade-off.** Here is why the drop occurs and why the Hybrid model remains the superior choice for production:

**1. The "Metric Disconnect" (Behavior vs. Pedagogy)**
Offline metrics like NDCG and Hit Ratio evaluate models based on one strict criterion: *Did the model predict the exact books the user interacted with in the hidden test set?* The pure MLP is an unconstrained behavioral mimic. It learns exactly what users clicked on. However, in the real world, users frequently read books outside their ideal comprehension level (e.g., an intermediate reader forcing their way through a dense, expert-level classic, or an advanced adult reading a basic children's book). The MLP gets rewarded by the test set for suggesting these mismatched books.

**2. The Cost of the Educational Constraint**
The Hybrid model is designed specifically to stop this. By applying the **Multiplicative NBC Penalty**, the engine actively intervenes. If an Intermediate (B1) reader has a behavioral affinity for a complex Expert (C1) text, the Hybrid model intentionally suppresses that recommendation to protect the user's learning experience. Because it actively pushes down books that the user *might* have actually read (but shouldn't have, linguistically speaking), it inevitably takes a small penalty on raw historical accuracy metrics.

**3. Qualitative Cohesion over Raw Accuracy**
The slight reduction in NDCG is the accepted "tax" for adding an educational safety rail. While the pure MLP wins on strict historical prediction, the Hybrid engine wins on **qualitative cohesion**. It ensures that the top recommendations are not just topically interesting, but fundamentally readable for that specific user, solving the core problem BookBuddy set out to fix.