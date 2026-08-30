"""
Decision Trees visualization functions for ML fundamentals notebook.

This module contains plotting functions extracted from the decision trees notebook
to reduce code-heavy appearance while preserving core teaching content.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ipywidgets import interact, IntSlider


def plot_pizza_preferences_3d(pizza_data, seed=42):
    """Plot pizza preferences in 3D scatter plot.

    ``likes_pizza`` is mapped to a *string* so that Plotly treats it as a
    category and honours the discrete colour map (a numeric column would
    silently get a continuous colour bar instead). ``likes_cheese`` and
    ``vegetarian`` are binary, so a little jitter is added to stop every
    point from collapsing onto four vertical lines.
    """
    rng = np.random.RandomState(seed)
    plot_df = pizza_data.copy()
    plot_df['Likes Pizza'] = np.where(plot_df['likes_pizza'] == 1, 'Yes 🍕', 'No 😞')

    for col in ('likes_cheese', 'vegetarian'):
        plot_df[col] = plot_df[col] + rng.uniform(-0.12, 0.12, size=len(plot_df))

    fig = px.scatter_3d(plot_df,
                        x='age', y='likes_cheese', z='vegetarian',
                        color='Likes Pizza',
                        # Colourblind-safe (blue / orange) rather than red / green.
                        color_discrete_map={'Yes 🍕': '#1f77b4', 'No 😞': '#ff7f0e'},
                        category_orders={'Likes Pizza': ['Yes 🍕', 'No 😞']},
                        title="🍕 Pizza Preferences in 3D (binary axes jittered)")

    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(height=500)
    fig.show()


def create_interactive_tree_builder(X_train, y_train, X_test, y_test, feature_cols=None):
    """Interactive decision-tree builder scored on held-out data.

    The widget deliberately reports *test* accuracy alongside training
    accuracy: judging a tree on the data it memorised would reward exactly
    the overfitting this section is about.
    """
    baseline = max(y_test.mean(), 1 - y_test.mean())

    @interact(
        max_depth=IntSlider(value=3, min=1, max=20, step=1, description='Max Depth'),
        min_samples_leaf=IntSlider(value=5, min=1, max=50, step=1, description='Min Leaf Samples'),
        min_samples_split=IntSlider(value=2, min=2, max=50, step=1, description='Min Split Samples'),
        max_leaf_nodes=IntSlider(value=0, min=0, max=30, step=1, description='Max Leaf Nodes')
    )
    def build_custom_tree(max_depth, min_samples_leaf, min_samples_split, max_leaf_nodes):
        kwargs = {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "random_state": 42
        }
        # Handle 0 as "None" for max_leaf_nodes
        if max_leaf_nodes > 0:
            kwargs["max_leaf_nodes"] = max_leaf_nodes

        custom_tree = DecisionTreeClassifier(**kwargs)
        custom_tree.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, custom_tree.predict(X_train))
        test_acc = accuracy_score(y_test, custom_tree.predict(X_test))
        gap = train_acc - test_acc
        importance = custom_tree.feature_importances_

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            f"Tree (Depth: {custom_tree.get_depth()}, Leaves: {custom_tree.get_n_leaves()}) | "
            f"Train: {train_acc:.1%}  Test: {test_acc:.1%}  Gap: {gap:+.1%}"
        )

        # --- Train vs test accuracy, against the majority-class baseline ---
        axs[0].barh(["Test", "Train"], [test_acc * 100, train_acc * 100],
                    color=['#1f77b4', '#aec7e8'])
        axs[0].axvline(baseline * 100, color='black', linestyle='--', linewidth=1.5)
        axs[0].text(baseline * 100 + 0.6, -0.42, f"baseline {baseline:.1%}",
                    fontsize=8, rotation=90, va='bottom')
        axs[0].set_xlim(0, 100)
        axs[0].set_title("Accuracy: train vs. held-out test")
        axs[0].set_xlabel("Percent")
        axs[0].grid(True, axis='x', linestyle='--', alpha=0.5)

        # --- Feature importance bar plot ---
        cols = feature_cols if feature_cols is not None else list(X_train.columns)
        axs[1].bar(cols, importance, color='teal')
        axs[1].set_title("Feature Importance")
        axs[1].set_ylabel("Importance")
        axs[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        # Feedback rewards *generalisation*, not memorisation.
        if gap > 0.05:
            print(f"\n⚠️ Overfitting: the tree scores {gap:.1%} higher on data it has "
                  f"already seen. Try a smaller depth or larger min-leaf.")
        elif test_acc <= baseline + 0.005:
            print(f"\n🤔 This tree barely beats always guessing the majority class "
                  f"({baseline:.1%}). It is probably underfitting — try a bit more depth.")
        else:
            print(f"\n✅ Nicely balanced: {test_acc:.1%} on unseen data "
                  f"({test_acc - baseline:+.1%} vs. baseline) with only a {gap:.1%} train/test gap.")

def plot_decision_tree_structure(tree, feature_cols):
    """Plot the structure of a decision tree."""
    plt.figure(figsize=(12, 6))
    plot_tree(tree,
              feature_names=feature_cols,
              class_names=['Dislike', 'Like'],
              filled=True, rounded=True)
    plt.show()


def plot_feature_importance_boxplots(forest, X):
    """Plot feature importance distributions across all trees in a random forest."""
    all_importances = np.array([
        tree.feature_importances_ for tree in forest.estimators_
    ])

    # Create box plots for each feature
    fig = go.Figure()

    for i, feature in enumerate(X.columns):
        fig.add_trace(go.Box(
            y=all_importances[:, i],
            name=feature,
            boxmean='sd',
            marker_color='teal'
        ))

    fig.update_layout(
        title="📊 Feature Importance Across Trees (Random Forest)",
        yaxis_title="Feature Importance",
        height=400
    )

    fig.show()


def plot_depth_vs_accuracy(X_train, X_test, y_train, y_test, max_depth_range=range(1, 15)):
    """Plot how tree depth affects training and test accuracy."""
    depths = max_depth_range
    train_accs, test_accs = [], []

    for d in depths:
        model = DecisionTreeClassifier(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, model.predict(X_train)))
        test_accs.append(accuracy_score(y_test, model.predict(X_test)))

    baseline = max(y_test.mean(), 1 - y_test.mean())
    best_depth = list(depths)[int(np.argmax(test_accs))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(depths), y=train_accs, mode='lines+markers', name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=list(depths), y=test_accs, mode='lines+markers', name='Test Accuracy'))
    fig.add_hline(y=baseline, line_dash="dot", line_color="grey",
                  annotation_text=f"always-guess-majority baseline ({baseline:.1%})",
                  annotation_position="bottom right")
    fig.add_vline(x=best_depth, line_dash="dash", line_color="green",
                  annotation_text=f"best test depth = {best_depth}",
                  annotation_position="top left")
    fig.update_layout(title="Effect of Tree Depth on Accuracy", xaxis_title="Max Depth", yaxis_title="Accuracy")
    fig.show()


def plot_model_comparison(results, baseline=None):
    """Plot comparison of different models' accuracies.

    ``baseline`` (the majority-class accuracy) is drawn as a reference line:
    without it, a bar chart of 64%-67% looks far more impressive than it is.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(results.keys()), y=list(results.values()),
                         text=[f"{v:.1%}" for v in results.values()],
                         textposition='auto', marker_color=["green", "blue", "orange"]))

    if baseline is not None:
        fig.add_hline(y=baseline, line_dash="dot", line_color="black",
                      annotation_text=f"always-guess-majority baseline ({baseline:.1%})",
                      annotation_position="bottom right")

    fig.update_layout(title="📊 Model Comparison on Pizza Preference",
                      yaxis_title="Accuracy", xaxis_title="Model",
                      yaxis_range=[0, 1],
                      height=400)
    fig.show()


def plot_boosting_performance(bdt, X_train, X_test, y_train, y_test):
    """Plot how gradient boosting performance evolves over boosting rounds."""
    train_errors = []
    test_errors = []

    for y_train_pred, y_test_pred in zip(
            bdt.staged_predict(X_train),
            bdt.staged_predict(X_test)):
        train_errors.append(1 - accuracy_score(y_train, y_train_pred))
        test_errors.append(1 - accuracy_score(y_test, y_test_pred))

    # Plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=train_errors,
        mode='lines+markers',
        name='Train Error',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        y=test_errors,
        mode='lines+markers',
        name='Test Error',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="📉 BDT Performance over Boosting Rounds",
        xaxis_title="Boosting Round",
        yaxis_title="Error Rate",
        height=400,
        legend=dict(x=0.7, y=0.95)
    )

    fig.show()


def generate_pizza_data(n_samples, seed=42):
    """Generate synthetic pizza preference dataset.

    Uses a *local* RandomState so that calling this function does not reseed
    NumPy's global generator behind the caller's back.
    """
    rng = np.random.RandomState(seed)

    # --- Core features ---
    ages = rng.randint(5, 70, size=n_samples)
    likes_cheese = rng.binomial(1, 0.75, size=n_samples)      # Most like cheese
    vegetarian = rng.binomial(1, 0.3, size=n_samples)         # Minority vegetarian
    has_pet = rng.binomial(1, 0.5, size=n_samples)            # Pure noise, on purpose

    # --- Extra features ---
    num_siblings = rng.poisson(1.5, size=n_samples)           # Adds noise & pattern
    favorite_topping = rng.choice(['pepperoni', 'mushroom', 'pineapple'], size=n_samples)

    # One-hot encode topping (as 0/1 ints so the table reads consistently)
    topping_dummies = pd.get_dummies(favorite_topping, prefix='topping').astype(int)

    # --- Calculate probability of liking pizza ---
    prob = np.zeros(n_samples)

    # Age-based base probability
    prob += np.select(
        [
            ages < 18,
            (ages >= 18) & (ages < 30),
            (ages >= 30) & (ages < 50),
            ages >= 50
        ],
        [0.70, 0.60, 0.50, 0.40]
    )

    # Add/subtract effects
    prob += 0.15 * likes_cheese
    prob -= 0.12 * vegetarian
    prob += 0.10 * (likes_cheese & (favorite_topping == 'mushroom'))   # mushroom-lovers
    prob -= 0.08 * (favorite_topping == 'pineapple')                  # 🍍 controversy!
    prob += 0.05 * (num_siblings >= 2)
    prob -= 0.05 * ((ages > 45) & (vegetarian == 1))                  # older vegetarians

    # Add interaction bonus
    interaction = ((ages < 25) & (vegetarian == 1) & (likes_cheese == 1)) * 0.15
    prob += interaction

    # Add some random noise
    prob += rng.normal(0, 0.05, size=n_samples)

    # Clip to valid probability range
    prob = np.clip(prob, 0, 1)

    # Final target variable. Note this is a *coin flip* with probability `prob`,
    # so even a perfect model cannot reach 100% accuracy -- the label itself is
    # noisy. This is why the baseline comparison in the notebook matters.
    likes_pizza = rng.binomial(1, prob)

    # --- Build DataFrame ---
    pizza_data = pd.DataFrame({
        'age': ages,
        'likes_cheese': likes_cheese,
        'vegetarian': vegetarian,
        'has_pet': has_pet,
        'num_siblings': num_siblings,
        'topping': favorite_topping,
    })

    # Add one-hot toppings
    pizza_data = pd.concat([pizza_data, topping_dummies], axis=1)
    feature_cols = ['age', 'likes_cheese', 'vegetarian', 'has_pet', 'num_siblings'] + \
                    [col for col in pizza_data.columns if col.startswith('topping_')]

    # Add target variable
    pizza_data['likes_pizza'] = likes_pizza

    return pizza_data, feature_cols


def gini(y):
    """Gini impurity of a set of labels: 0 = pure, 0.5 = maximally mixed (2 classes)."""
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1.0 - (p ** 2 + (1 - p) ** 2)


def plot_gini_split_search(X, y, feature='age'):
    """Show *how* a tree picks its best question, using Gini impurity.

    Scans every candidate threshold for one feature, plots the weighted
    impurity of the two resulting groups, and marks the winning split.
    """
    y = np.asarray(y)
    values = np.asarray(X[feature])
    parent = gini(y)

    candidates = np.unique(values)
    candidates = (candidates[:-1] + candidates[1:]) / 2  # midpoints between values

    thresholds, weighted, gains = [], [], []
    for t in candidates:
        left, right = y[values <= t], y[values > t]
        if len(left) == 0 or len(right) == 0:
            continue
        w = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
        thresholds.append(t)
        weighted.append(w)
        gains.append(parent - w)

    best = int(np.argmin(weighted))
    best_t, best_w, best_gain = thresholds[best], weighted[best], gains[best]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=weighted, mode='lines+markers',
                             name='Impurity after split'))
    fig.add_hline(y=parent, line_dash="dot", line_color="grey",
                  annotation_text=f"impurity before any split ({parent:.4f})",
                  annotation_position="top left")
    fig.add_vline(x=best_t, line_dash="dash", line_color="green",
                  annotation_text=f"best: {feature} ≤ {best_t:g}",
                  annotation_position="bottom right")
    fig.update_layout(
        title=f"🔎 Searching for the best question on '{feature}'",
        xaxis_title=f"Threshold for '{feature}'",
        yaxis_title="Weighted Gini impurity of the two groups",
        height=420
    )
    fig.show()

    print(f"Impurity before splitting:      {parent:.4f}")
    print(f"Best question found:            is {feature} <= {best_t:g}?")
    print(f"Impurity after that split:      {best_w:.4f}")
    print(f"Information gain:               {best_gain:.4f}")
    print("\n📋 Best possible gain, feature by feature (this is how the tree ranks questions):")
    for col in X.columns:
        vals = np.asarray(X[col])
        cand = np.unique(vals)
        if len(cand) < 2:
            continue
        cand = (cand[:-1] + cand[1:]) / 2
        best_col = max(
            (parent - (len(y[vals <= t]) * gini(y[vals <= t])
                       + len(y[vals > t]) * gini(y[vals > t])) / len(y), t)
            for t in cand
            if 0 < len(y[vals <= t]) < len(y)
        )
        print(f"   {col:<20} gain={best_col[0]:.4f}  (at {col} <= {best_col[1]:g})")
