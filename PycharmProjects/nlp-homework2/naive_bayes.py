"""
naive_bayes.py
--------------
A complete Naive Bayes text classifier built from scratch.
No external libraries used.
"""

import math

# ─────────────────────────────────────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────────────────────────────────────

dataset = []
with open("/Users/radio/Downloads/IMDB Dataset.csv") as f:
    next(f)  # skip header row
    for line in f:
        parts = line.strip().rsplit(",", 1)
        if len(parts) == 2:
            text, label = parts[0], parts[1]
            dataset.append((text, label))

# Split into train (80%) and test (20%)
split = int(len(dataset) * 0.8)
train_set = dataset[:split]
test_set  = dataset[split:]

print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")

# ─────────────────────────────────────────────────────────────────────────────
# Tokenization & Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text):
    """Converts a string into a list of lowercase tokens."""
    return text.lower().split()


def build_vocab(dataset):
    """Returns a sorted list of all unique words across all texts."""
    vocab = set()
    for text, label in dataset:
        for token in tokenize(text):
            vocab.add(token)
    return sorted(vocab)


# ─────────────────────────────────────────────────────────────────────────────
# Count words per class  (train only!)
# ─────────────────────────────────────────────────────────────────────────────

total_words  = {"positive": 0, "negative": 0}
class_counts = {"positive": 0, "negative": 0}
word_counts  = {"positive": {}, "negative": {}}

for text, label in train_set:
    tokens = tokenize(text)
    class_counts[label] += 1
    total_words[label]  += len(tokens)
    for token in tokens:
        if token not in word_counts[label]:
            word_counts[label][token] = 0
        word_counts[label][token] += 1

vocab = build_vocab(train_set)
print(f"Vocab size: {len(vocab)}")

# ─────────────────────────────────────────────────────────────────────────────
# Priors  P(class)
# ─────────────────────────────────────────────────────────────────────────────

total_docs = sum(class_counts.values())

priors = {}
for label, count in class_counts.items():
    priors[label] = count / total_docs

print("Priors:", priors)

# ─────────────────────────────────────────────────────────────────────────────
# Likelihoods  P(word | class)  with Laplace smoothing
# using log probabilities to avoid underflow with long texts
# ─────────────────────────────────────────────────────────────────────────────

log_likelihoods = {"positive": {}, "negative": {}}

for label in ["positive", "negative"]:
    for word in vocab:
        count       = word_counts[label].get(word, 0)
        numerator   = count + 1
        denominator = total_words[label] + len(vocab)
        log_likelihoods[label][word] = math.log(numerator / denominator)

# ─────────────────────────────────────────────────────────────────────────────
# Scoring  (using log sums instead of products to avoid underflow)
# ─────────────────────────────────────────────────────────────────────────────

def score(text, label):
    """Returns the log Naive Bayes score for a text belonging to a class."""
    tokens = tokenize(text)
    result = math.log(priors[label])
    for token in tokens:
        if token in log_likelihoods[label]:
            result += log_likelihoods[label][token]
        else:
            result += math.log(1 / (total_words[label] + len(vocab)))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(text):
    """Returns the class with the highest Naive Bayes score."""
    scores = {label: score(text, label) for label in priors}
    return max(scores, key=scores.get)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nRunning predictions on test set (this may take a minute)...")

    y_true = [label for text, label in test_set]
    y_pred = [predict(text) for text, _ in test_set]

    from evaluation import Evaluator
    ev = Evaluator(y_true, y_pred)
    ev.report()