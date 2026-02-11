import pandas as pd
import os
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "data/training_master.xlsx"   # must exist
MODEL_DIR = "models"

TEXT_COL_KEYWORDS = [
    "company name",
    "company domain",
    "domain",
    "website",
    "company website",
    "linkedin",
    "company linkedin",
    "company linkedin url",
    "company type",
]

INDUSTRY_COL_KEYWORDS = [
    "industry",
    "company industry",
]

MIN_ROWS_PER_CLASS = 25

# ============================================================
# HELPERS
# ============================================================

def clean_text(txt: str) -> str:
    txt = str(txt).lower()
    txt = re.sub(r"http\\S+", " ", txt)
    txt = re.sub(r"www\\S+", " ", txt)
    txt = re.sub(r"[^a-z0-9 ]+", " ", txt)
    txt = re.sub(r"\\s+", " ", txt)
    return txt.strip()


def find_column(cols, keywords):
    for c in cols:
        for k in keywords:
            if k in c:
                return c
    return None


# ============================================================
# LOAD DATA
# ============================================================

print("📥 Loading training dataset...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"❌ Training file not found: {DATA_FILE}\n"
        "Create data/training_master.xlsx first."
    )

if DATA_FILE.endswith(".csv"):
    df = pd.read_csv(DATA_FILE, encoding="latin1")
else:
    df = pd.read_excel(DATA_FILE)

df.columns = [c.strip().lower() for c in df.columns]

print("Columns found:")
print(df.columns.tolist())

industry_col = find_column(df.columns, INDUSTRY_COL_KEYWORDS)

if not industry_col:
    raise ValueError("❌ No industry column found for training!")

text_cols = [
    c for c in df.columns
    if any(k in c for k in TEXT_COL_KEYWORDS)
]

if not text_cols:
    raise ValueError("❌ No usable text columns found!")

print("🏷️ Industry column:", industry_col)
print("📝 Text columns:", text_cols)

# ============================================================
# FILTER + CLEAN
# ============================================================

df = df.dropna(subset=[industry_col])
df[industry_col] = df[industry_col].astype(str)

counts = df[industry_col].value_counts()

print("\nClass counts BEFORE filtering:")
print(counts)

good_classes = counts[counts >= MIN_ROWS_PER_CLASS].index

df = df[df[industry_col].isin(good_classes)]

print("\nClasses kept:", len(good_classes))
print(good_classes.tolist())

# ---- FORCE STRINGS BEFORE JOIN ----

df[text_cols] = df[text_cols].astype(str)

df["__text__"] = df[text_cols].fillna("").agg(" ".join, axis=1)
df["__text__"] = df["__text__"].apply(clean_text)

print("\nTotal training rows:", len(df))

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["__text__"],
    df[industry_col],
    test_size=0.15,
    random_state=42,
    stratify=df[industry_col],
)

# ============================================================
# TF-IDF
# ============================================================

print("\n🧠 Training TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=4,
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ============================================================
# CLASSIFIER
# ============================================================

print("\n🤖 Training classifier...")

clf = LogisticRegression(
    max_iter=2500,
    n_jobs=-1,
)

clf.fit(X_train_vec, y_train)

# ============================================================
# EVALUATION
# ============================================================

preds = clf.predict(X_test_vec)

print("\n📊 Evaluation report:\n")
print(classification_report(y_test, preds))

# ============================================================
# SAVE MODELS
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

tfidf_path = os.path.join(MODEL_DIR, "industry_tfidf.joblib")
clf_path = os.path.join(MODEL_DIR, "industry_ml_model.joblib")

joblib.dump(tfidf, tfidf_path)
joblib.dump(clf, clf_path)

print("\n✅ MODELS SAVED:")
print(tfidf_path)
print(clf_path)
