"""
Benchmark helpers for ``05_agentic_ai_llms.ipynb``.

That notebook asks: *we have the actual data sets from Class 1 and Class 2, and
the actual models we trained on them — so how does a language model score on the
very same benchmark?* The sentiment data and the small parsers live next door in
``llm_simple.py``; the screening benchmark, the scoring and the charts live here.

As in the other notebooks, the data plumbing, the plots and the scoring live
here so the notebook itself stays short. **The prompting stays in the
notebook**, because the prompting is the teaching content.

Everything in this module runs offline. Only the notebook calls the API — with
one exception: :func:`run_in_parallel`, which merely *organises* the calls the
notebook hands it.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .classification import generate_screening_data, make_classifier

# Same palette as the other notebooks, so a colour means the same thing across
# the whole course.
BLUE, ORANGE, GREEN, PURPLE, GREY = '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#7f7f7f'
LIGHT_BLUE = '#9ecae1'


# ---------------------------------------------------------------------------
# 1. Making many API calls without waiting all afternoon
# ---------------------------------------------------------------------------

def run_in_parallel(fn, items, workers=8, retries=1, show_progress=True):
    """Apply ``fn`` to every item, several calls at a time, keeping the order.

    One API call takes a second or two. Fifty of them, one after another, is a
    minute and a half of a classroom staring at a progress bar — so we send a
    handful at once. Nothing clever happens here: it is the same calls in the
    same order, just overlapped.

    ``fn(item)`` returning ``None`` counts as a failure (a reply we could not
    parse, or a network hiccup) and is retried ``retries`` times before we give
    up on that item and leave a ``None`` in the results.
    """
    items = list(items)
    results = [None] * len(items)

    def attempt(index):
        for tries in range(retries + 1):
            try:
                value = fn(items[index])
            except Exception:                     # a dropped connection, a 429…
                value = None
            if value is not None:
                return index, value
            time.sleep(0.5 * (tries + 1))         # back off a little, then retry
        return index, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(attempt, i) for i in range(len(items))]
        for future in as_completed(futures):
            index, value = future.result()
            results[index] = value
            if show_progress:
                print("·" if value is not None else "✗", end="", flush=True)

    if show_progress:
        failed = sum(v is None for v in results)
        print(f"  ({len(items)} calls" + (f", {failed} unusable)" if failed else ")"))
    return results


# ---------------------------------------------------------------------------
# 2. Class 1's data, Class 1's model, and a benchmark set we can afford
# ---------------------------------------------------------------------------

class ScreeningBenchmark:
    """Class 1's screening study, packaged for a like-for-like comparison.

    The data, the train/test split and the classifier are *exactly* the ones
    from ``01_classification.ipynb`` — same generator, same seed, same 70/30
    stratified split, same logistic regression. The only new thing is
    ``patients``: a small sample of the held-out test set that we can afford to
    send to a language model one row at a time.

    That sample is drawn **balanced** (half with the disease, half without)
    rather than at the natural 10% prevalence. With 10% prevalence, 50 patients
    would contain about 5 real cases, and recall measured on 5 cases is not a
    measurement. The price of that choice: accuracy on this set is *not*
    comparable to the 92.6% in Class 1 — here the always-say-healthy baseline
    scores 50%, not 90%. Every model in the notebook is scored on these same
    rows, so the comparison between them is still fair.
    """

    def __init__(self, n_benchmark=50, seed=7):
        self.data, self.feature_cols = generate_screening_data(n_samples=10_000)
        X = self.data[self.feature_cols]
        y = self.data['has_disease']

        # Class 1's split, reproduced exactly.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        self.model = make_classifier().fit(self.X_train, self.y_train)

        # How the trained model does on the *whole* test set, for reference.
        full_risk = self.model.predict_proba(self.X_test)[:, 1]
        self.full_test_accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
        self.full_test_auc = roc_auc_score(self.y_test, full_risk)
        self.prevalence = float(y.mean())

        # The benchmark rows: half sick, half healthy, all from the test set.
        rng = np.random.RandomState(seed)
        sick = self.y_test[self.y_test == 1].index.to_numpy()
        well = self.y_test[self.y_test == 0].index.to_numpy()
        n_sick = n_benchmark // 2
        chosen = np.concatenate([
            rng.choice(sick, n_sick, replace=False),
            rng.choice(well, n_benchmark - n_sick, replace=False),
        ])
        rng.shuffle(chosen)

        self.patients = self.data.loc[chosen].reset_index(drop=True)
        self.truth = self.patients['has_disease'].to_numpy()

        # What Class 1's model says about these same patients.
        features = self.patients[self.feature_cols]
        self.model_risk = self.model.predict_proba(features)[:, 1]
        self.model_pred = (self.model_risk >= 0.5).astype(int)
        self.baseline_pred = np.zeros(len(self.truth), dtype=int)   # "always healthy"

        # Class 1's model was trained where 10% of patients are ill and is asked
        # here about a set where 50% are. At the default 0.5 threshold it
        # therefore under-flags badly -- not because it is a bad model, but
        # because the question changed underneath it. Moving the threshold to
        # the training prevalence undoes exactly that shift, and is the dial
        # Class 1 Part 6 spent a whole section on.
        self.tuned_threshold = round(self.prevalence, 2)
        self.model_pred_tuned = (self.model_risk >= self.tuned_threshold).astype(int)

        # Four patients -- two ill, two not, in a shuffled order -- for the
        # "guess it yourself before the model does" exercise. The selection
        # happens here rather than in the notebook so that reading the notebook
        # cell does not give the answers away.
        ill = rng.choice(np.where(self.truth == 1)[0], 2, replace=False)
        healthy = rng.choice(np.where(self.truth == 0)[0], 2, replace=False)
        self.quiz_index = np.concatenate([ill, healthy])
        rng.shuffle(self.quiz_index)

    def describe(self):
        """Print what we are about to benchmark on."""
        n = len(self.patients)
        print(f"📋 Class 1's study: {len(self.data):,} patients, "
              f"{self.prevalence:.1%} of them with the disease.")
        print(f"   Trained on {len(self.X_train):,}, held out {len(self.X_test):,}.")
        print(f"   Class 1's model on the full held-out set: "
              f"accuracy {self.full_test_accuracy:.1%}, AUC {self.full_test_auc:.3f}")
        print()
        print(f"🎯 Our benchmark: {n} of those held-out patients "
              f"({int(self.truth.sum())} with the disease, {int(n - self.truth.sum())} without).")
        print(f"   Deliberately balanced, so 'always say healthy' scores 50% here, not 90%.")
        print(f"   Every approach in this notebook is judged on these same {n} people.")
        print(f"   Because of that, Class 1's model also gets its threshold moved from 0.50 "
              f"to {self.tuned_threshold:.2f}, to match this sample.")


def load_screening_benchmark(n_benchmark=50, seed=7):
    """Build the screening benchmark. See :class:`ScreeningBenchmark`."""
    return ScreeningBenchmark(n_benchmark=n_benchmark, seed=seed)


def patient_to_text(row):
    """Turn one row of the patient table into a sentence a model can read.

    This is the step with no equivalent in Class 1. The classifier took six
    numbers; a language model takes English, so somebody has to decide how to
    say "marker_a = 4.12" out loud — and that wording is now part of the model.
    """
    return (f"Age: {row.age} years. "
            f"BMI: {row.bmi}. "
            f"Close relative has had the disease: {'yes' if row.family_history else 'no'}. "
            f"Current smoker: {'yes' if row.smoker else 'no'}. "
            f"Blood marker A: {row.marker_a}. "
            f"Blood marker B: {row.marker_b}.")


def labelled_examples(bench, n=30, seed=0):
    """A block of solved examples from the *training* set, as text.

    These are patients Class 1's model was allowed to learn from, so handing
    them to the language model is fair play: both approaches now see labelled
    data from the same source. Drawn balanced, to match the benchmark set.
    """
    rng = np.random.RandomState(seed)
    sick = bench.y_train[bench.y_train == 1].index.to_numpy()
    well = bench.y_train[bench.y_train == 0].index.to_numpy()
    chosen = np.concatenate([
        rng.choice(sick, n // 2, replace=False),
        rng.choice(well, n - n // 2, replace=False),
    ])
    rng.shuffle(chosen)

    lines = []
    for row in bench.data.loc[chosen].itertuples():
        answer = "DISEASE" if row.has_disease else "HEALTHY"
        lines.append(f"{patient_to_text(row)} -> {answer}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Scoring — the Class 1 metrics, on whatever produced the predictions
# ---------------------------------------------------------------------------

def wilson_interval(correct, total, z=1.96):
    """95% confidence interval for an accuracy, the way Class 1 should have.

    A small benchmark gives a shaky number, and the honest way to report one is
    with the range it could plausibly have come from. Wilson's version behaves
    sensibly near 0% and 100%, where the textbook formula does not.
    """
    if total == 0:
        return (float('nan'), float('nan'))
    p = correct / total
    denominator = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_classification(truth, predicted, risk=None, missing_as=0):
    """Accuracy, precision, recall, F1 (and AUC if given risk scores).

    ``predicted`` may contain ``None`` for replies we could not parse. Those
    are counted, then filled in with ``missing_as`` (0 = "healthy") rather than
    dropped, so every approach is scored on the identical set of patients. A
    real system needs exactly this decision, and it is never free.
    """
    truth = np.asarray(truth)
    predicted = list(predicted)
    n_missing = sum(p is None for p in predicted)
    filled = np.array([missing_as if p is None else int(p) for p in predicted])

    tn, fp, fn, tp = confusion_matrix(truth, filled, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    correct = int(tp + tn)
    low, high = wilson_interval(correct, len(truth))

    scores = {
        'accuracy': correct / len(truth),
        'accuracy_low': low,
        'accuracy_high': high,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'caught': int(tp),
        'missed': int(fn),
        'false_alarms': int(fp),
        'unusable_replies': n_missing,
    }
    if risk is not None:
        clean = np.array([0.5 if r is None else float(r) for r in risk])
        scores['auc'] = roc_auc_score(truth, clean) if len(set(truth)) > 1 else float('nan')
    return scores


def show_scoreboard(results, title="Scoreboard"):
    """Print one row per approach: the Class 1 metrics, side by side."""
    rows = []
    for name, s in results.items():
        row = {
            'Approach': name,
            'Accuracy': f"{s['accuracy']:.0%}",
            '95% CI': f"{s['accuracy_low']:.0%}–{s['accuracy_high']:.0%}",
            'Precision': f"{s['precision']:.0%}",
            'Recall': f"{s['recall']:.0%}",
            'F1': f"{s['f1']:.2f}",
            'Caught': s['caught'],
            'Missed': s['missed'],
            'False alarms': s['false_alarms'],
        }
        if any('auc' in scores for scores in results.values()):
            row['AUC'] = f"{s['auc']:.3f}" if 'auc' in s else "—"
        rows.append(row)

    table = pd.DataFrame(rows)
    print(f"🏁 {title}\n")
    if _in_notebook():
        from IPython.display import display
        display(table.style.hide(axis="index"))
    else:
        print(table.to_string(index=False))
    return table


def plot_metric_comparison(results, metrics=('accuracy', 'precision', 'recall', 'f1'),
                           title="Every approach, on the same patients", colours=None):
    """Grouped bars: one group per metric, one bar per approach.

    The default palette assumes the notebook's ordering: baseline first in
    grey, the two trained-model rows in two blues, the language model in warm
    colours. Pass ``colours`` to override.
    """
    fig = go.Figure()
    colours = colours or [GREY, BLUE, LIGHT_BLUE, ORANGE, GREEN, PURPLE]
    for (name, scores), colour in zip(results.items(), colours):
        values = [scores.get(m, 0.0) * 100 for m in metrics]
        fig.add_trace(go.Bar(
            name=name,
            x=[m.capitalize() for m in metrics],
            y=values,
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            marker_color=colour,
        ))
    fig.update_layout(
        title=f"📊 {title}",
        barmode='group',
        yaxis_title="Score (%)", yaxis_range=[0, 112],
        height=480,
        legend=dict(orientation='h', y=-0.18),
    )
    fig.show()


# ---------------------------------------------------------------------------
# 4. Class 2's houses — an honest test on the ten houses we already have
# ---------------------------------------------------------------------------

def leave_one_out_line(sizes, prices):
    """Class 2's straight line, tested on houses it never saw.

    Ten houses is not enough for a train/test split, so we do the next best
    thing: hide one house, fit the line on the other nine, predict the hidden
    one, and repeat ten times. Every prediction is then made by a line that has
    never seen the house it is predicting — the same handicap the language
    model has, which is the only way this comparison means anything.
    """
    sizes = np.asarray(sizes, dtype=float)
    prices = np.asarray(prices, dtype=float)
    predictions = np.empty(len(sizes))

    for i in range(len(sizes)):
        keep = np.arange(len(sizes)) != i
        line = LinearRegression().fit(sizes[keep].reshape(-1, 1), prices[keep])
        predictions[i] = line.predict([[sizes[i]]])[0]
    return predictions


def other_houses_as_text(sizes, prices, hide_index):
    """The nine houses that are *not* house ``hide_index``, written out.

    This is the language model's version of "fit on the other nine": we cannot
    train it, so we paste the nine sales into the question instead.
    """
    lines = [f"{size:,} sq ft sold for ${price}k"
             for i, (size, price) in enumerate(zip(sizes, prices)) if i != hide_index]
    return "\n".join(lines)


def score_regression(truth, guesses):
    """Average miss, worst miss and RMSE, ignoring replies with no number."""
    pairs = [(t, g) for t, g in zip(truth, guesses) if g is not None]
    skipped = len(truth) - len(pairs)
    if not pairs:
        return {'mae': float('nan'), 'rmse': float('nan'),
                'worst': float('nan'), 'skipped': skipped}
    actual = np.array([t for t, _ in pairs], dtype=float)
    predicted = np.array([g for _, g in pairs], dtype=float)
    return {
        'mae': float(mean_absolute_error(actual, predicted)),
        'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        'worst': float(np.max(np.abs(actual - predicted))),
        'skipped': skipped,
    }


def plot_price_comparison(sizes, prices, series,
                          title="Ten houses, three ways to guess the price"):
    """Real prices as dots, plus one marker series per approach."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=prices, mode="markers", name="What it actually sold for",
        marker=dict(size=15, color="#2c3e50", symbol="circle"),
    ))
    styles = [(BLUE, "line"), (ORANGE, "x"), (GREEN, "diamond"), (PURPLE, "square")]
    for (name, values), (colour, symbol) in zip(series.items(), styles):
        pairs = [(s, v) for s, v in zip(sizes, values) if v is not None]
        if symbol == "line":
            fig.add_trace(go.Scatter(
                x=[s for s, _ in pairs], y=[v for _, v in pairs],
                mode="lines+markers", name=name,
                line=dict(color=colour, width=3),
                marker=dict(size=8, color=colour),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[s for s, _ in pairs], y=[v for _, v in pairs],
                mode="markers", name=name,
                marker=dict(size=13, color=colour, symbol=symbol),
            ))
    fig.update_layout(
        title=f"🏠 {title}",
        xaxis_title="Size (square feet)",
        yaxis_title="Price (thousands of dollars)",
        height=520,
        legend=dict(orientation='h', y=-0.2),
    )
    fig.show()


def plot_mae_bars(results, title="Average miss per house"):
    """Bar chart of the average error, in dollars. Shorter is better."""
    names = list(results)
    values = [results[n]['mae'] * 1000 for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=values,
        text=[f"${v:,.0f}" for v in values], textposition="outside",
        marker_color=[BLUE, ORANGE, GREEN, PURPLE][:len(names)],
    ))
    fig.update_layout(
        title=f"📏 {title} — lower is better",
        yaxis_title="Average miss (dollars)",
        yaxis_range=[0, max(values) * 1.25 if values else 1],
        showlegend=False, height=420,
    )
    fig.show()


# ---------------------------------------------------------------------------
# 5. What it costs, once you multiply it by a real workload
# ---------------------------------------------------------------------------

def cost_projection(usage, predictions_per_day=10_000, seconds_per_call=None):
    """Take the running bill so far and scale it up to a real workload.

    ``usage`` is the ``USAGE`` dict from ``llm_client`` — passed in rather than
    imported, so this module never touches the API itself.
    """
    calls = usage.get('calls', 0)
    if not calls:
        print("No API calls have been made yet, so there is nothing to scale up.")
        return None

    per_call = usage.get('cost', 0.0) / calls
    tokens_per_call = (usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)) / calls

    print(f"📊 {calls} calls so far, {tokens_per_call:,.0f} tokens each on average.")
    print(f"   That is ${per_call:.5f} per prediction.\n")
    print(f"💸 At {predictions_per_day:,} predictions a day:")
    print(f"   Language model: ${per_call * predictions_per_day:,.2f} per day "
          f"= ${per_call * predictions_per_day * 365:,.0f} per year")
    print(f"   Class 1's logistic regression: essentially $0 "
          f"(it is six multiplications and an addition)")

    if seconds_per_call:
        hours = seconds_per_call * predictions_per_day / 3600
        print(f"\n⏱️  And {seconds_per_call:.1f}s per call means "
              f"{hours:,.1f} hours of waiting per day if you run them one at a time.")

    return per_call


def plot_cost_scaleup(cost_per_call, volumes=(100, 1_000, 10_000, 100_000, 1_000_000)):
    """The per-prediction fee, drawn at volumes people actually operate at."""
    fig = go.Figure(go.Bar(
        x=[f"{v:,}/day" for v in volumes],
        y=[cost_per_call * v * 365 for v in volumes],
        text=[f"${cost_per_call * v * 365:,.0f}" for v in volumes],
        textposition="outside", marker_color=ORANGE,
    ))
    fig.update_layout(
        title="💸 Yearly bill for the language model, by volume",
        yaxis_title="Dollars per year", yaxis_type="log",
        showlegend=False, height=420,
    )
    fig.show()


def _in_notebook():
    """True only inside a real Jupyter/IPython frontend that can render HTML."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False
