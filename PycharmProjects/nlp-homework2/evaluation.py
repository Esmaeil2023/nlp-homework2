"""
evaluation.py
-------------
Evaluation metrics for binary and multi-class classification.
No external libraries used.
"""


class Evaluator:
    """
    Computes evaluation metrics for binary and multi-class classification.    Binary mode  : labels are 0 / 1
    Multi-class  : labels are any hashable values (ints, strings, …)

    Usage
    -----
    ev = Evaluator(y_true, y_pred)

    # binary helpers
    ev.confusion_matrix()   -> (TP, FP, FN, TN)
    ev.accuracy()           -> float
    ev.precision()          -> float
    ev.recall()             -> float
    ev.f1_score()           -> float

    # multi-class (also works for binary)
    ev.confusion_matrix_multiclass()          -> dict[label -> dict[label -> int]]
    ev.precision_per_class()                  -> dict[label -> float]
    ev.recall_per_class()                     -> dict[label -> float]
    ev.f1_per_class()                         -> dict[label -> float]
    ev.macro_precision()                      -> float
    ev.macro_recall()                         -> float
    ev.macro_f1()                             -> float
    ev.weighted_precision()                   -> float
    ev.weighted_recall()                      -> float
    ev.weighted_f1()                          -> float
    ev.report()                               -> prints a formatted summary
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, y_true: list, y_pred: list):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if len(y_true) == 0:
            raise ValueError("y_true and y_pred must not be empty.")

        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = sorted(set(y_true) | set(y_pred))

    # ------------------------------------------------------------------
    # Binary metrics  (positive class = 1)
    # ------------------------------------------------------------------

    def confusion_matrix(self):
        """
        Binary confusion matrix.
        Returns: (TP, FP, FN, TN)
        """
        TP = sum(1 for yt, yp in zip(self.y_true, self.y_pred) if yt == 1 and yp == 1)
        FP = sum(1 for yt, yp in zip(self.y_true, self.y_pred) if yt == 0 and yp == 1)
        FN = sum(1 for yt, yp in zip(self.y_true, self.y_pred) if yt == 1 and yp == 0)
        TN = sum(1 for yt, yp in zip(self.y_true, self.y_pred) if yt == 0 and yp == 0)
        return TP, FP, FN, TN

    def accuracy(self) -> float:
        """Fraction of correct predictions."""
        correct = sum(1 for yt, yp in zip(self.y_true, self.y_pred) if yt == yp)
        return correct / len(self.y_true)

    def precision(self) -> float:
        """Binary precision. Returns 0.0 on division-by-zero."""
        TP, FP, _, _ = self.confusion_matrix()
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def recall(self) -> float:
        """Binary recall. Returns 0.0 on division-by-zero."""
        TP, _, FN, _ = self.confusion_matrix()
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def f1_score(self) -> float:
        """Binary F1. Returns 0.0 on division-by-zero."""
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # ------------------------------------------------------------------
    # Multi-class confusion matrix
    # ------------------------------------------------------------------

    def confusion_matrix_multiclass(self) -> dict:
        """
        Returns a nested dict: matrix[true_label][pred_label] = count.
        Rows = actual class, Columns = predicted class.
        """
        matrix = {c: {c2: 0 for c2 in self.classes} for c in self.classes}
        for yt, yp in zip(self.y_true, self.y_pred):
            matrix[yt][yp] += 1
        return matrix

    # ------------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------------

    def precision_per_class(self) -> dict:
        """
        For each class c:  TP_c / (TP_c + FP_c)
        FP_c = everything predicted as c that is NOT c.
        """
        matrix = self.confusion_matrix_multiclass()
        result = {}
        for c in self.classes:
            tp = matrix[c][c]
            fp = sum(matrix[other][c] for other in self.classes if other != c)
            result[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return result

    def recall_per_class(self) -> dict:
        """
        For each class c:  TP_c / (TP_c + FN_c)
        FN_c = actual c predicted as something else.
        """
        matrix = self.confusion_matrix_multiclass()
        result = {}
        for c in self.classes:
            tp = matrix[c][c]
            fn = sum(matrix[c][other] for other in self.classes if other != c)
            result[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return result

    def f1_per_class(self) -> dict:
        """Per-class F1 score."""
        prec = self.precision_per_class()
        rec  = self.recall_per_class()
        result = {}
        for c in self.classes:
            p, r = prec[c], rec[c]
            result[c] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return result

    # ------------------------------------------------------------------
    # Macro averages  (unweighted mean over classes)
    # ------------------------------------------------------------------

    def macro_precision(self) -> float:
        vals = self.precision_per_class().values()
        return sum(vals) / len(self.classes)

    def macro_recall(self) -> float:
        vals = self.recall_per_class().values()
        return sum(vals) / len(self.classes)

    def macro_f1(self) -> float:
        vals = self.f1_per_class().values()
        return sum(vals) / len(self.classes)

    # ------------------------------------------------------------------
    # Weighted averages  (weighted by support = number of true samples)
    # ------------------------------------------------------------------

    def _support(self) -> dict:
        """Number of true samples per class."""
        return {c: sum(1 for yt in self.y_true if yt == c) for c in self.classes}

    def weighted_precision(self) -> float:
        support = self._support()
        total   = sum(support.values())
        prec    = self.precision_per_class()
        return sum(prec[c] * support[c] for c in self.classes) / total if total > 0 else 0.0

    def weighted_recall(self) -> float:
        support = self._support()
        total   = sum(support.values())
        rec     = self.recall_per_class()
        return sum(rec[c] * support[c] for c in self.classes) / total if total > 0 else 0.0

    def weighted_f1(self) -> float:
        support = self._support()
        total   = sum(support.values())
        f1      = self.f1_per_class()
        return sum(f1[c] * support[c] for c in self.classes) / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Pretty report
    # ------------------------------------------------------------------

    def report(self):
        """Print a classification report."""
        support = self._support()
        prec    = self.precision_per_class()
        rec     = self.recall_per_class()
        f1      = self.f1_per_class()

        col = 12
        header = f"{'class':<{col}}{'precision':>{col}}{'recall':>{col}}{'f1-score':>{col}}{'support':>{col}}"
        divider = "-" * len(header)

        print("\n=== Classification Report ===")
        print(divider)
        print(header)
        print(divider)
        for c in self.classes:
            print(
                f"{str(c):<{col}}"
                f"{prec[c]:>{col}.4f}"
                f"{rec[c]:>{col}.4f}"
                f"{f1[c]:>{col}.4f}"
                f"{support[c]:>{col}}"
            )
        print(divider)
        total = sum(support.values())
        print(f"{'accuracy':<{col}}{'':>{col}}{'':>{col}}{self.accuracy():>{col}.4f}{total:>{col}}")
        print(f"{'macro avg':<{col}}{self.macro_precision():>{col}.4f}{self.macro_recall():>{col}.4f}{self.macro_f1():>{col}.4f}{total:>{col}}")
        print(f"{'weighted avg':<{col}}{self.weighted_precision():>{col}.4f}{self.weighted_recall():>{col}.4f}{self.weighted_f1():>{col}.4f}{total:>{col}}")
        print(divider)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def _assert_close(a, b, tol=1e-6, msg=""):
    assert abs(a - b) < tol, f"FAIL {msg}: expected {b}, got {a}"


def test_binary_basic():
    """From class: y_true=[1,0,0,1,0], y_pred=[1,0,0,0,1]"""
    ev = Evaluator([1, 0, 0, 1, 0], [1, 0, 0, 0, 1])
    assert ev.confusion_matrix() == (1, 1, 1, 2), "binary CM"
    _assert_close(ev.accuracy(),  0.6,  msg="accuracy")
    _assert_close(ev.precision(), 0.5,  msg="precision")
    _assert_close(ev.recall(),    0.5,  msg="recall")
    _assert_close(ev.f1_score(),  0.5,  msg="f1")
    print("PASS  test_binary_basic")


def test_binary_perfect():
    """Perfect predictor."""
    ev = Evaluator([1, 0, 1, 0], [1, 0, 1, 0])
    _assert_close(ev.accuracy(),  1.0, msg="accuracy")
    _assert_close(ev.precision(), 1.0, msg="precision")
    _assert_close(ev.recall(),    1.0, msg="recall")
    _assert_close(ev.f1_score(),  1.0, msg="f1")
    print("PASS  test_binary_perfect")


def test_binary_all_wrong():
    """Fully inverted predictor — triggers every division-by-zero path."""
    ev = Evaluator([1, 1, 0, 0], [0, 0, 1, 1])
    assert ev.confusion_matrix() == (0, 2, 2, 0), "inverted CM"
    _assert_close(ev.accuracy(),  0.0, msg="accuracy")
    _assert_close(ev.precision(), 0.0, msg="precision")
    _assert_close(ev.recall(),    0.0, msg="recall")
    _assert_close(ev.f1_score(),  0.0, msg="f1")
    print("PASS  test_binary_all_wrong")


def test_binary_all_positive_predicted():
    """Model always predicts 1."""
    ev = Evaluator([1, 0, 1, 0], [1, 1, 1, 1])
    _assert_close(ev.precision(), 0.5,  msg="precision")
    _assert_close(ev.recall(),    1.0,  msg="recall")
    _assert_close(ev.f1_score(),  2/3,  msg="f1")
    print("PASS  test_binary_all_positive_predicted")


def test_multiclass_basic():
    """
    3-class problem: cats=0, dogs=1, birds=2
    Hand-computed expected values.
    """
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2, 0, 2, 2]
    ev = Evaluator(y_true, y_pred)

    _assert_close(ev.accuracy(), 6/9, msg="mc accuracy")

    prec = ev.precision_per_class()
    rec  = ev.recall_per_class()
    f1   = ev.f1_per_class()

    # class 0: TP=2, FP=1(bird->0), FN=1(cat->dog)  => P=2/3, R=2/3
    _assert_close(prec[0], 2/3, msg="prec class 0")
    _assert_close(rec[0],  2/3, msg="rec  class 0")

    # class 1: TP=2, FP=1(cat->dog), FN=1(dog->bird) => P=2/3, R=2/3
    _assert_close(prec[1], 2/3, msg="prec class 1")
    _assert_close(rec[1],  2/3, msg="rec  class 1")

    # class 2: TP=2, FP=1(dog->bird), FN=1(bird->cat) => P=2/3, R=2/3
    _assert_close(prec[2], 2/3, msg="prec class 2")
    _assert_close(rec[2],  2/3, msg="rec  class 2")

    _assert_close(ev.macro_precision(), 2/3, msg="macro prec")
    _assert_close(ev.macro_recall(),    2/3, msg="macro rec")
    _assert_close(ev.macro_f1(),        2/3, msg="macro f1")

    # balanced classes → weighted == macro
    _assert_close(ev.weighted_f1(), ev.macro_f1(), msg="weighted==macro")
    print("PASS  test_multiclass_basic")


def test_multiclass_perfect():
    y_true = [0, 1, 2, 0, 1, 2]
    ev = Evaluator(y_true, y_true)
    _assert_close(ev.accuracy(),        1.0, msg="mc perfect acc")
    _assert_close(ev.macro_f1(),        1.0, msg="mc perfect macro f1")
    _assert_close(ev.weighted_f1(),     1.0, msg="mc perfect weighted f1")
    print("PASS  test_multiclass_perfect")


def test_multiclass_division_by_zero():
    """Class 2 never appears in predictions → precision for class 2 = 0."""
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]   # class 2 never predicted
    ev = Evaluator(y_true, y_pred)
    prec = ev.precision_per_class()
    rec  = ev.recall_per_class()
    _assert_close(prec[2], 0.0, msg="unseen pred precision")
    _assert_close(rec[2],  0.0, msg="unseen pred recall")
    print("PASS  test_multiclass_division_by_zero")


def test_multiclass_string_labels():
    """Labels can be strings."""
    y_true = ["cat", "dog", "cat", "bird", "dog"]
    y_pred = ["cat", "cat", "cat", "bird", "dog"]
    ev = Evaluator(y_true, y_pred)
    _assert_close(ev.accuracy(), 4/5, msg="string labels accuracy")
    print("PASS  test_multiclass_string_labels")


def run_all_tests():
    print("=" * 50)
    print("Running all tests …")
    print("=" * 50)
    test_binary_basic()
    test_binary_perfect()
    test_binary_all_wrong()
    test_binary_all_positive_predicted()
    test_multiclass_basic()
    test_multiclass_perfect()
    test_multiclass_division_by_zero()
    test_multiclass_string_labels()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all_tests()

    print()
    print("--- Binary example (from class) ---")
    ev = Evaluator([1, 0, 0, 1, 0], [1, 0, 0, 0, 1])
    TP, FP, FN, TN = ev.confusion_matrix()
    print(f"  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"  Accuracy : {ev.accuracy():.4f}")
    print(f"  Precision: {ev.precision():.4f}")
    print(f"  Recall   : {ev.recall():.4f}")
    print(f"  F1 Score : {ev.f1_score():.4f}")

    print()
    print("--- Multi-class example (3 classes) ---")
    ev3 = Evaluator(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 0, 1, 1, 1, 2, 0, 2, 2],
    )
    ev3.report()

