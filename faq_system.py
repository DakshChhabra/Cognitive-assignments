
ROLL_NO = "1024170069"
# ------------------------------------------------------------

import re
import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)

# ============================================================
# Q1: Personalized Knowledge Base
# ============================================================
print("=" * 70)
print("Q1: Personalized Knowledge Base")
print("=" * 70)

fixed_entries = [
    {"question": "what is the annual fee", "answer": "The annual fee is Rs 500.",
     "keywords": "fee cost price charge", "category": "billing"},
    {"question": "how to reset password", "answer": "Go to Settings > Reset Password.",
     "keywords": "password reset login", "category": "account"},
    {"question": "what are your working hours", "answer": "We are open 9 AM to 5 PM.",
     "keywords": "hours timing open time", "category": "general"},
    {"question": "how can i pay the fee", "answer": "You can pay via UPI, card, or net banking.",
     "keywords": "pay payment upi fee", "category": "billing"},
]

pool = {
    "billing": [
        {"question": "how do i get a refund for a duplicate transaction",
         "answer": "Raise a refund request under Billing > Refunds; it is processed in 5-7 working days.",
         "keywords": "refund duplicate transaction money"},
        {"question": "where can i download my invoice",
         "answer": "Go to Billing > History and click Download Invoice next to the payment.",
         "keywords": "invoice receipt download bill"},
    ],
    "account": [
        {"question": "how do i update my registered mobile number",
         "answer": "Go to Profile > Contact Details, enter the new number and verify it with the OTP.",
         "keywords": "update mobile number phone"},
        {"question": "how do i delete my account",
         "answer": "Go to Settings > Privacy > Delete Account and confirm with your password.",
         "keywords": "delete close deactivate account"},
    ],
    "general": [
        {"question": "where is your office located",
         "answer": "Our office is on the main campus, Block A, Ground Floor.",
         "keywords": "office location address visit"},
        {"question": "how can i contact customer support",
         "answer": "Email support@example.com or call 1800-000-000 during working hours.",
         "keywords": "contact support helpline email"},
    ],
}

categories = ["billing", "account", "general"]
digits = [int(c) for c in ROLL_NO if c.isdigit()]
last_two = digits[-2:]

personal_entries = []
used = {"billing": 0, "account": 0, "general": 0}
for d in last_two:
    cat = categories[d % 3]
    entry = dict(pool[cat][used[cat] % len(pool[cat])])
    entry["category"] = cat
    used[cat] += 1
    personal_entries.append(entry)
    print(f"digit {d} -> category[{d} % 3] = {cat}")

df = pd.DataFrame(fixed_entries + personal_entries,
                  columns=["question", "answer", "keywords", "category"])
print("\nFinal 6-row DataFrame:\n")
print(df)

# ============================================================
# Q2: Generate and Score a Hypothesis
# ============================================================
STOPWORDS = {"what", "is", "the", "how", "to", "can", "i", "do", "my", "are",
             "your", "a", "an", "of", "for", "where", "me", "you", "does", "in"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOPWORDS]


def score_query(query, data):
    """Return all matching entries ranked by confidence (highest first).

    Scoring: each query word found in an entry's keywords = 2 points,
    found only in the question text = 1 point.
    Confidence = score / max possible score (2 points per query word).
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return data.iloc[0:0].assign(score=[], confidence=[])
    rows = []
    for idx, row in data.iterrows():
        kw = set(tokenize(row["keywords"]))
        qs = set(tokenize(row["question"]))
        score = sum(2 if t in kw else (1 if t in qs else 0) for t in q_tokens)
        if score > 0:
            rows.append({"index": idx, "question": row["question"], "answer": row["answer"],
                         "category": row["category"], "score": score,
                         "confidence": round(score / (2 * len(q_tokens)), 2)})
    result = pd.DataFrame(rows, columns=["index", "question", "answer", "category",
                                         "score", "confidence"])
    return result.sort_values("score", ascending=False).reset_index(drop=True)


print("\n" + "=" * 70)
print("Q2: Scoring function demo")
print("=" * 70)
for q in ["how to reset my password", "annual fee"]:
    print(f"\nQuery: '{q}'")
    print(score_query(q, df)[["question", "category", "score", "confidence"]])

# ============================================================
# Q3: same_category
# ============================================================


def same_category(category_name, data):
    """Return all questions belonging to the given category."""
    return data[data["category"] == category_name]["question"].tolist()


print("\n" + "=" * 70)
print("Q3: Questions in a personalized entry's category")
print("=" * 70)
cat_to_check = personal_entries[0]["category"]
print(f"Category: {cat_to_check}")
for q in same_category(cat_to_check, df):
    print(" -", q)

# ============================================================
# Q4: Add a keyword and save to CSV
# ============================================================
print("\n" + "=" * 70)
print("Q4: Add a keyword to an entry and save CSV")
print("=" * 70)
target = 0  # entry: "what is the annual fee"
print(f"Selected entry: '{df.loc[target, 'question']}'")
print(f"Current keywords: {df.loc[target, 'keywords']}")
new_kw = input("Enter a new keyword to add: ").strip().lower()
if new_kw and new_kw not in df.loc[target, "keywords"].split():
    df.loc[target, "keywords"] += " " + new_kw
filename = f"{ROLL_NO}_faq_data.csv"
df.to_csv(filename, index=False)
print(f"Updated keywords: {df.loc[target, 'keywords']}")
print(f"Saved DataFrame to {filename}")
print(df)

# ============================================================
# Q5: groupby count per category
# ============================================================
print("\n" + "=" * 70)
print("Q5: FAQ entries per category")
print("=" * 70)
print(df.groupby("category").size().rename("count"))

# ============================================================
# Q6: Tie-aware scoring
# ============================================================


def best_matches(query, data):
    """Print the best match; if several tie for the top score, print all of them."""
    ranked = score_query(query, data)
    if ranked.empty:
        print("No matching entry found.")
        return ranked
    top = ranked["score"].max()
    best = ranked[ranked["score"] == top]
    if len(best) > 1:
        print(f"TIE: {len(best)} entries share the highest score ({top}):")
    else:
        print(f"Single best match (score {top}):")
    for _, r in best.iterrows():
        print(f"  Q: {r['question']}\n  A: {r['answer']}\n  "
              f"[category: {r['category']}, confidence: {r['confidence']}]\n")
    return best


print("\n" + "=" * 70)
print("Q6: Tie handling demo")
print("=" * 70)
print("\nQuery 1 (tie expected): 'fee'")
best_matches("fee", df)
print("Query 2 (no tie): 'how to reset password'")
best_matches("how to reset password", df)
