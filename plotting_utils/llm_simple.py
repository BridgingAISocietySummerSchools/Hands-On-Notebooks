"""
Helper functions for ``05_agentic_ai_simple.ipynb``.

That notebook asks one question: *can a language model do the classification and
regression jobs we built by hand in classes 1 and 2 — without any training?*

As in the other notebooks, the data sets, plots and widgets live here so that the
notebook itself stays short and readable. The prompting stays in the notebook,
because the prompting *is* the teaching content.

The plots and parsers in here run fully offline. Only the notebook calls the API.
"""

import re

import pandas as pd
import plotly.graph_objects as go
from ipywidgets import Text, Textarea, interact_manual


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

# Eight customer reviews with a "correct" answer we agreed on beforehand.
#
# The first five are easy. The last three are deliberately hard: sarcasm, and
# reviews that say something good *and* something bad. A room full of humans
# would not fully agree on those three either -- which is the point. When the
# model gets one "wrong", the first question to ask is whether the label was
# ever really right.
REVIEWS = [
    {"text": "The battery lasts all day and the screen is gorgeous. Best purchase this year.",
     "label": "positive"},
    {"text": "Arrived broken, and support never replied to any of my three emails.",
     "label": "negative"},
    {"text": "Does exactly what it promises. No complaints at all.",
     "label": "positive"},
    {"text": "Cheap plastic. It stopped working after two weeks.",
     "label": "negative"},
    {"text": "Setup took ten minutes and it has worked flawlessly ever since.",
     "label": "positive"},
    {"text": "Oh it's wonderful, if you enjoy reading a 60-page manual to turn on a lamp.",
     "label": "negative"},          # sarcasm — no negative word anywhere
    {"text": "The sound quality is excellent, but the app crashes every single day.",
     "label": "negative"},          # good point, then a dealbreaker
    {"text": "Shipping was painfully slow, but honestly the product is worth the wait.",
     "label": "positive"},          # bad point, then the verdict
]

# The same ten houses as in Class 2, so the comparison is like for like.
# Sizes in square feet, prices in thousands of dollars.
HOUSE_SIZES = [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
HOUSE_PRICES = [150, 185, 215, 230, 270, 295, 350, 415, 410, 435]


# ---------------------------------------------------------------------------
# 2. Turning a sentence of English into a value we can compute with
# ---------------------------------------------------------------------------
#
# A classical model returns a number. A language model returns *prose*. Getting
# from one to the other is real work, and it is work you will always have to do.

def to_label(reply, labels, default="unclear"):
    """Find which of ``labels`` a model's reply is saying.

    Tolerant on purpose: the model may answer "positive", "Positive.",
    or "This review is clearly positive." All three should count.
    """
    text = str(reply).lower()
    hits = [label for label in labels if label.lower() in text]
    # Exactly one label mentioned -> unambiguous. Zero or several -> give up
    # loudly rather than guess, so the notebook can show it as 'unclear'.
    return hits[0] if len(hits) == 1 else default


def to_number(reply):
    """Pull the first number out of a model's reply, or return None.

    Handles "350", "$350", "about 350k" and "1,200". Returns a float so the
    notebook can plot it; None means the reply had no number in it at all.
    """
    match = re.search(r"-?\d[\d,]*\.?\d*", str(reply))
    if not match:
        return None
    return float(match.group().replace(",", ""))


def to_price_in_thousands(reply):
    """Like ``to_number``, but for prices we asked for in thousands.

    Models sometimes answer "$350,000" when asked for "350". Both mean the same
    house, so we rescale — and say so, rather than silently plotting a point
    that is 1000x off.
    """
    value = to_number(reply)
    if value is None:
        return None
    if value > 5000:                       # answered in dollars, not thousands
        print(f"   (rescaled {value:,.0f} dollars to {value / 1000:.0f} thousand)")
        return value / 1000
    return value


def _in_notebook():
    """True only inside a real Jupyter/IPython frontend that can render HTML."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 3. Showing classification results
# ---------------------------------------------------------------------------

def show_label_results(texts, expected, predicted, max_chars=60):
    """Print a review-by-review table and return the accuracy.

    Returns the fraction correct, so the notebook can compare it to the
    accuracy of the trained classifier from Class 1.
    """
    rows = []
    for text, truth, guess in zip(texts, expected, predicted):
        short = text if len(text) <= max_chars else text[:max_chars - 1] + "…"
        rows.append({
            "Review": short,
            "Our label": truth,
            "Model said": guess,
            "": "✅" if guess == truth else "❌",
        })
    table = pd.DataFrame(rows)
    if _in_notebook():                     # nice HTML table in Jupyter
        from IPython.display import display
        display(table.style.hide(axis="index"))
    else:                                  # plain text anywhere else
        print(table.to_string(index=False))

    correct = sum(g == t for g, t in zip(predicted, expected))
    total = len(expected)
    print(f"\n🎯 The model agreed with us on {correct} of {total} reviews "
          f"({correct / total:.0%}) — with zero training examples.")
    return correct / total


def plot_accuracy_bars(scores, title="How accurate was each approach?"):
    """Compare accuracies of different approaches. ``scores`` is {name: 0..1}."""
    names = list(scores)
    values = [scores[n] * 100 for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        text=[f"{v:.0f}%" for v in values], textposition="outside",
        marker_color=["#5499c7", "#48c9b0", "#e59866", "#af7ac5"][:len(names)],
    ))
    fig.update_layout(
        title=f"🎯 {title}",
        yaxis_title="Accuracy (%)", yaxis_range=[0, 112],
        showlegend=False,
    )
    fig.show()


# ---------------------------------------------------------------------------
# 4. Showing regression results
# ---------------------------------------------------------------------------

def plot_llm_vs_line(sizes, actual, llm_guesses, line_predictions):
    """Real prices, the fitted straight line, and the model's guesses."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sizes, y=actual, mode="markers", name="Real price",
        marker=dict(size=13, color="#2c3e50", symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=line_predictions, mode="lines", name="Class 2: fitted line",
        line=dict(color="#5499c7", width=3),
    ))
    # Only plot the guesses the model actually gave a number for.
    pairs = [(s, g) for s, g in zip(sizes, llm_guesses) if g is not None]
    fig.add_trace(go.Scatter(
        x=[s for s, _ in pairs], y=[g for _, g in pairs],
        mode="markers", name="Language model's guess",
        marker=dict(size=13, color="#e59866", symbol="x"),
    ))
    fig.update_layout(
        title="🏠 Two ways to guess a house price",
        xaxis_title="Size (square feet)",
        yaxis_title="Price (thousands of dollars)",
    )
    fig.show()


def show_error_comparison(actual, llm_guesses, line_predictions):
    """Print the average miss for each approach, in dollars."""
    def average_miss(guesses):
        misses = [abs(g - a) for g, a in zip(guesses, actual) if g is not None]
        return sum(misses) / len(misses) if misses else float("nan")

    llm_miss = average_miss(llm_guesses)
    line_miss = average_miss(line_predictions)

    print("📏 On average, each guess was off by:")
    print(f"   Class 2's fitted line:  ${line_miss * 1000:,.0f}")
    print(f"   The language model:     ${llm_miss * 1000:,.0f}")
    skipped = sum(1 for g in llm_guesses if g is None)
    if skipped:
        print(f"   ({skipped} model reply/replies had no number in them and were skipped)")

    # Say who won out loud. Which one wins depends on the market the model
    # happens to assume, so the notebook must not hard-code an outcome.
    if line_miss < llm_miss:
        print("\n🏆 The fitted line was closer — on the data it was fitted to.")
    else:
        print("\n🏆 The language model was closer this time — worth asking why, "
              "and whether it holds if you run it again.")
    return {"line": line_miss, "llm": llm_miss}


def plot_repeat_spread(values, label="Asked the same question 5 times"):
    """Show how much the answers moved when we asked the identical question.

    A trained regression model gives the same answer every time. A language
    model, asked to be creative, does not — and that is worth *seeing*.
    """
    clean = [v for v in values if v is not None]
    fig = go.Figure(go.Scatter(
        x=clean, y=[1] * len(clean), mode="markers+text",
        text=[f"{v:.0f}" for v in clean], textposition="top center",
        marker=dict(size=16, color="#e59866"),
        showlegend=False,
    ))
    if clean:
        spread = max(clean) - min(clean)
        fig.update_layout(
            title=f"🎲 {label} — answers spanned ${spread * 1000:,.0f}",
        )
    fig.update_layout(
        xaxis_title="Predicted price (thousands of dollars)",
        yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
        height=280,
    )
    fig.show()


# ---------------------------------------------------------------------------
# 5. A playground, so people can try their own text
# ---------------------------------------------------------------------------

def create_classifier_playground(classify_fn,
                                 example_text="My train was 40 minutes late again.",
                                 example_labels="complaint, compliment, question"):
    """Type any text and any set of categories, and see what comes back.

    ``classify_fn(text, labels)`` takes a string and a list of label strings.

    Uses a *button* rather than live updating: every run costs a real API call,
    so nothing should fire while you are still typing.
    """
    print("💡 Change either box, then press 'Run'. Some category sets to try:")
    print("   • urgent, normal        • billing, technical, other")
    print("   • happy, angry, confused, neutral")
    print()

    @interact_manual(
        text=Textarea(value=example_text, description="Text:",
                      layout={"width": "80%", "height": "60px"}),
        categories=Text(value=example_labels, description="Categories:",
                        layout={"width": "80%"}),
    )
    def run(text, categories):
        labels = [c.strip() for c in categories.split(",") if c.strip()]
        if not text.strip() or len(labels) < 2:
            print("✋ Give me some text and at least two categories.")
            return
        print(f"🏷️  {classify_fn(text, labels)}")
