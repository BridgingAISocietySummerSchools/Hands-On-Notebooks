"""
Classification visualization functions for the classification notebook.

This module contains the synthetic screening dataset and the plotting functions
used by ``01_classification.ipynb``, extracted from the notebook to reduce its
code-heavy appearance while preserving the core teaching content.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from ipywidgets import interact, FloatSlider

# Colourblind-safe pair used throughout, matching the other notebooks.
BLUE, ORANGE, GREY = '#1f77b4', '#ff7f0e', '#7f7f7f'


def make_classifier():
    """The one model used throughout the notebook.

    Logistic regression behind a scaler: the scaler only matters for the
    optimiser's comfort and for reading coefficients on a common scale, but
    bundling them keeps every ``fit`` call in the notebook a one-liner.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def generate_screening_data(n_samples=10_000, seed=42):
    """Generate a synthetic medical-screening dataset.

    Follows the same conventions as ``generate_pizza_data`` in
    ``decision_trees.py``: a *local* RandomState so the caller's global seed is
    untouched, and a final **coin flip** for the label so that even a perfect
    model cannot reach 100%. The label is genuinely noisy, which is what makes
    the baseline comparison in the notebook worth making.

    Two properties are deliberate, because the notebook builds sections on them:

    * **Prevalence is about 10%.** Always predicting "no disease" therefore
      scores ~90% accuracy while catching nobody -- the point of Part 2.
    * **``site`` has no effect on the label at all**, but the three sites
      recruit very different age groups. Site A is a young-adult clinic, so a
      study run there alone never sees the ages where ``marker_a`` actually
      matters. That is the sampling-bias demonstration in Part 9.
    """
    rng = np.random.RandomState(seed)

    # --- Where each patient was recruited, and how old they are ---------------
    # The site itself is medically irrelevant. It only decides who walks in.
    site = rng.choice(['A', 'B', 'C'], size=n_samples, p=[0.34, 0.33, 0.33])
    age_centre = np.select([site == 'A', site == 'B', site == 'C'], [38.0, 54.0, 68.0])
    age = np.clip(rng.normal(age_centre, 6.0), 20, 90).round().astype(int)

    # --- Everything else measured at the appointment -------------------------
    bmi = np.clip(rng.normal(26.5 + 0.03 * (age - 50), 4.0), 16, 45).round(1)
    family_history = rng.binomial(1, 0.18, size=n_samples)
    smoker = rng.binomial(1, 0.24, size=n_samples)

    # marker_a is the expensive blood test, marker_b the cheap one.
    marker_a = np.clip(rng.normal(3.0 + 0.02 * (age - 50) + 0.6 * family_history, 1.0), 0, None).round(2)
    marker_b = np.clip(rng.normal(1.0 + 0.25 * smoker, 0.6), 0, None).round(2)

    # --- How likely each patient is to actually have the disease -------------
    # Written on the log-odds scale purely so the overall prevalence can be set
    # with a single intercept; the notebook never shows or needs this formula.
    age_z = (age - 50) / 15.0
    marker_a_z = marker_a - 3.0
    marker_b_z = (marker_b - 1.0) / 0.6
    bmi_z = (bmi - 26.5) / 4.0

    log_odds = (
        -4.1                                        # sets prevalence to ~10%
        + 0.45 * age_z
        + 0.50 * bmi_z
        + 1.00 * family_history
        + 0.70 * smoker
        + 0.30 * marker_a_z
        + 0.40 * marker_b_z
        # The marker only really bites in older patients. A study that recruits
        # only young adults cannot learn this -- which is the whole point of
        # the sampling-bias section.
        + 1.30 * marker_a_z * np.maximum(age_z, 0)
    )
    probability = 1.0 / (1.0 + np.exp(-log_odds))

    # The coin flip: two identical patients can still get different outcomes.
    has_disease = rng.binomial(1, probability)

    data = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'family_history': family_history,
        'smoker': smoker,
        'marker_a': marker_a,
        'marker_b': marker_b,
        'site': site,
        'has_disease': has_disease,
    })

    feature_cols = ['age', 'bmi', 'family_history', 'smoker', 'marker_a', 'marker_b']
    return data, feature_cols


def plot_class_balance(y):
    """Show how lopsided the two classes are, in counts and in percent."""
    y = np.asarray(y)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())

    fig = go.Figure(go.Bar(
        x=['Healthy (0)', 'Has disease (1)'],
        y=[n_neg, n_pos],
        text=[f"{n_neg:,}<br>{n_neg / len(y):.1%}", f"{n_pos:,}<br>{n_pos / len(y):.1%}"],
        textposition='auto',
        marker_color=[BLUE, ORANGE],
    ))
    fig.update_layout(
        title=f"⚖️ Class balance: {n_pos / len(y):.1%} of {len(y):,} patients have the disease",
        yaxis_title="Number of patients",
        height=380,
    )
    fig.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix"):
    """Draw a 2x2 confusion matrix with each cell named in plain language.

    sklearn's ``confusion_matrix`` returns rows = truth, columns = prediction,
    ordered [0, 1]. The rows are reversed for display so that "has disease"
    sits at the top, which is how clinical 2x2 tables are usually drawn.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Top row = actually diseased, bottom row = actually healthy.
    counts = [[fn, tp], [tn, fp]]
    labels = [
        [f"<b>{fn:,}</b><br>missed cases<br>(false negatives)",
         f"<b>{tp:,}</b><br>caught<br>(true positives)"],
        [f"<b>{tn:,}</b><br>correctly cleared<br>(true negatives)",
         f"<b>{fp:,}</b><br>false alarms<br>(false positives)"],
    ]
    # Colour by "is this cell a mistake?", not by size -- the big true-negative
    # block would otherwise dominate the scale and hide everything else.
    is_error = [[1, 0], [0, 1]]

    fig = go.Figure(go.Heatmap(
        z=is_error,
        x=["Model says: healthy", "Model says: disease"],
        y=["Actually has disease", "Actually healthy"],
        text=labels,
        texttemplate="%{text}",
        colorscale=[[0, '#d9ead3'], [1, '#f4cccc']],
        showscale=False,
        hovertemplate="%{y}<br>%{x}<extra></extra>",
    ))
    fig.update_layout(
        title=f"🔲 {title}   (n = {int(np.sum(counts)):,})",
        height=380,
        xaxis=dict(side='top'),
    )
    fig.show()
    return tn, fp, fn, tp


def plot_metric_bars(metrics, baseline=None, title="How the model scores"):
    """Compare several metrics side by side.

    ``metrics`` maps a metric name to a value in [0, 1]. ``baseline`` may map
    the same names to the always-say-no model's scores, drawn alongside -- the
    contrast is the entire argument of the section.
    """
    fig = go.Figure()
    names = list(metrics.keys())

    if baseline is not None:
        fig.add_trace(go.Bar(
            x=names, y=[baseline[k] for k in names], name="Always say 'healthy'",
            text=[f"{baseline[k]:.2f}" for k in names], textposition='auto',
            marker_color=GREY, opacity=0.65,
        ))

    fig.add_trace(go.Bar(
        x=names, y=[metrics[k] for k in names], name="Our model",
        text=[f"{metrics[k]:.2f}" for k in names], textposition='auto',
        marker_color=BLUE,
    ))

    fig.update_layout(
        title=f"📊 {title}",
        yaxis_title="Score", yaxis_range=[0, 1.05],
        barmode='group', height=420,
    )
    fig.show()


def _metrics_at(y_true, y_scores, threshold):
    """Precision, recall and F1 for the positive class at one threshold.

    Returns zeros rather than NaNs when the model flags nobody, so the widget
    stays readable when the slider is pushed to 1.0.
    """
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average='binary', zero_division=0
    )
    return y_pred, precision, recall, f1


def create_threshold_interactive(y_true, y_scores):
    """Move the decision threshold and watch the four cells and three metrics move.

    Deliberately shows the confusion matrix and the metrics together: the point
    is that precision and recall are not independent dials, they are two views
    of the same four numbers being pushed around by one choice.
    """
    y_true = np.asarray(y_true)
    n_positive = int(y_true.sum())

    @interact(threshold=FloatSlider(value=0.5, min=0.02, max=0.98, step=0.02,
                                    description='Threshold', continuous_update=False))
    def show(threshold):
        y_pred, precision, recall, f1 = _metrics_at(y_true, y_scores, threshold)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        fig.suptitle(f"Flag a patient when the predicted risk is at least {threshold:.2f}")

        axs[0].bar(["Caught\n(TP)", "Missed\n(FN)", "False alarms\n(FP)"],
                   [tp, fn, fp], color=[BLUE, ORANGE, GREY])
        axs[0].set_ylabel("Patients")
        axs[0].set_title(f"{tp + fp:,} patients flagged for follow-up")
        axs[0].grid(True, axis='y', linestyle='--', alpha=0.4)

        axs[1].bar(["Precision", "Recall", "F1"], [precision, recall, f1],
                   color=[BLUE, ORANGE, GREY])
        axs[1].set_ylim(0, 1.05)
        axs[1].set_title("Precision / recall / F1")
        axs[1].grid(True, axis='y', linestyle='--', alpha=0.4)
        for i, v in enumerate([precision, recall, f1]):
            axs[1].text(i, v + 0.03, f"{v:.2f}", ha='center')

        plt.tight_layout()
        plt.show()

        print(f"Of {n_positive:,} patients who really have the disease, "
              f"this threshold catches {tp:,} and misses {fn:,}.")
        if tp + fp == 0:
            print("⚠️ Nobody is flagged at all. Precision is undefined and recall is zero — "
                  "the model has become the always-say-'healthy' baseline.")
        elif recall > 0.9:
            print("🔍 Screening mode: almost nobody slips through, but look at the false alarms.")
        elif precision > 0.6:
            print("🎯 Confirmation mode: a flag now means something, but look at how many cases are missed.")


def plot_threshold_sweep(y_true, y_scores):
    """Precision and recall as a function of the threshold, on one pair of axes.

    The static companion to the slider: it shows the *whole* trade-off at once,
    including that the two curves never peak in the same place.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.argmax(f1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precision, mode='lines',
                             name='Precision', line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=recall, mode='lines',
                             name='Recall', line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=f1, mode='lines',
                             name='F1', line=dict(color=GREY, width=2, dash='dot')))
    fig.add_vline(x=thresholds[best], line_dash="dash", line_color=GREY,
                  annotation_text=f"best F1 at {thresholds[best]:.2f}",
                  annotation_position="top right")
    fig.add_vline(x=0.5, line_dash="dot", line_color="black",
                  annotation_text="the default 0.50", annotation_position="bottom left")
    fig.update_layout(
        title="⚖️ One dial, two metrics pulling in opposite directions",
        xaxis_title="Decision threshold", yaxis_title="Score",
        yaxis_range=[0, 1.02], height=430,
    )
    fig.show()
    return thresholds[best]


def plot_pr_curve(curves, prevalence):
    """Precision-recall curves, with the always-guess-positive floor drawn in.

    ``curves`` maps a label to ``(y_true, y_scores)``. Under heavy imbalance the
    prevalence line is the honest zero point: a useless model sits on it, and a
    ROC curve would not show that.
    """
    fig = go.Figure()
    colours = [BLUE, ORANGE, GREY]

    for (name, (y_true, y_scores)), colour in zip(curves.items(), colours):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=name,
                                 line=dict(color=colour, width=3)))

    fig.add_hline(y=prevalence, line_dash="dot", line_color="black",
                  annotation_text=f"flag everyone ({prevalence:.1%} precision)",
                  annotation_position="bottom right")
    fig.update_layout(
        title="📉 Precision-recall curve",
        xaxis_title="Recall (fraction of real cases caught)",
        yaxis_title="Precision (fraction of flags that are real)",
        xaxis_range=[0, 1], yaxis_range=[0, 1.02], height=460,
    )
    fig.show()


def plot_roc_curves(curves, title="ROC curves"):
    """Overlay ROC curves and report each AUC.

    ``curves`` maps a label to ``(y_true, y_scores)``. The diagonal -- a model
    that has learned nothing -- is always drawn, because AUC only means
    something relative to it.
    """
    fig = go.Figure()
    colours = [BLUE, ORANGE, '#2ca02c', '#9467bd']
    aucs = {}

    for (name, (y_true, y_scores)), colour in zip(curves.items(), colours):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        aucs[name] = auc
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f"{name} — AUC {auc:.3f}",
                                 line=dict(color=colour, width=3)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name="Random guessing — AUC 0.500",
                             line=dict(color='black', width=2, dash='dot')))
    fig.update_layout(
        title=f"📈 {title}",
        xaxis_title="False positive rate (healthy people we alarm)",
        yaxis_title="True positive rate (sick people we catch)",
        xaxis_range=[0, 1], yaxis_range=[0, 1.02], height=520,
        legend=dict(x=0.4, y=0.12, bgcolor='rgba(255,255,255,0.7)'),
    )
    fig.show()
    return aucs


def plot_split_lottery(X, y, n_repeats=30, test_size=0.3, cv_scores=None):
    """Refit on many random train/test splits and show how far the AUC wanders.

    If ``cv_scores`` is given (from ``cross_val_score``), its mean is drawn as a
    line so the notebook can make the point that cross-validation reports one
    number where a single split reports a lottery ticket.
    """
    aucs = []
    for seed in range(n_repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
        model = make_classifier().fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

    aucs = np.array(aucs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, n_repeats + 1), y=aucs, mode='markers',
        name="One random split", marker=dict(color=BLUE, size=9),
    ))
    fig.add_hline(y=aucs.mean(), line_dash="dash", line_color=BLUE,
                  annotation_text=f"mean of {n_repeats} splits: {aucs.mean():.3f}",
                  annotation_position="top left")
    if cv_scores is not None:
        fig.add_hline(y=float(np.mean(cv_scores)), line_dash="solid", line_color=ORANGE,
                      annotation_text=f"5-fold cross-validation: {np.mean(cv_scores):.3f}",
                      annotation_position="bottom left")
    fig.update_layout(
        title=(f"🎲 The same model, {n_repeats} different splits: "
               f"AUC ranges from {aucs.min():.3f} to {aucs.max():.3f}"),
        xaxis_title="Split number (nothing changed but the random seed)",
        yaxis_title="AUC on that split's test set",
        height=430, showlegend=False,
    )
    fig.show()
    return aucs


def plot_kfold_diagram(n_splits=5):
    """The five-practice-exams picture: which slice is held out on each round."""
    fig = go.Figure()

    for fold in range(n_splits):
        for block in range(n_splits):
            held_out = (block == fold)
            fig.add_shape(
                type="rect",
                x0=block, x1=block + 0.94,
                y0=-fold, y1=-fold + 0.8,
                fillcolor=ORANGE if held_out else BLUE,
                opacity=1.0 if held_out else 0.35,
                line=dict(color="white", width=2),
            )
            fig.add_annotation(
                x=block + 0.47, y=-fold + 0.4,
                text="test" if held_out else "train",
                showarrow=False, font=dict(color="white", size=11),
            )
        fig.add_annotation(x=-0.15, y=-fold + 0.4, text=f"Round {fold + 1}",
                           showarrow=False, xanchor="right", font=dict(size=12))

    fig.update_layout(
        title=f"🔁 {n_splits}-fold cross-validation: every patient is tested on exactly once",
        xaxis=dict(visible=False, range=[-1.4, n_splits]),
        yaxis=dict(visible=False, range=[-n_splits + 0.1, 1.1]),
        height=300, margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='white',
    )
    fig.show()


def plot_bias_demo(results, title="Deployed on patients aged 55+"):
    """Compare two models' AUC, where the only difference is who was recruited.

    ``results`` maps a label to an AUC. Kept deliberately plain: the whole
    argument is that two bars differ although the algorithm never changed.
    """
    names = list(results.keys())
    values = [results[k] for k in names]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        text=[f"AUC {v:.3f}" for v in values], textposition='auto',
        marker_color=[ORANGE if i == 0 else BLUE for i in range(len(names))],
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="black",
                  annotation_text="random guessing", annotation_position="bottom right")
    fig.update_layout(
        title=f"🧭 {title}",
        yaxis_title="AUC", yaxis_range=[0.4, 1.0], height=420,
    )
    fig.show()
