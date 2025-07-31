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
from ipywidgets import interact, IntSlider


def plot_pizza_preferences_3d(pizza_data):
    """Plot pizza preferences in 3D scatter plot."""
    fig = px.scatter_3d(pizza_data,
                        x='age', y='likes_cheese', z='vegetarian',
                        color='likes_pizza',
                        color_discrete_map={0: 'red', 1: 'green'},
                        title="🍕 Pizza Preferences in 3D",
                        labels={'likes_pizza': 'Likes Pizza'})

    fig.update_layout(height=500)
    fig.show()


def create_interactive_tree_builder(X, y, feature_cols):
    """Create interactive widget for building custom decision trees."""
    @interact(
        max_depth=IntSlider(value=3, min=1, max=10, step=1, description='Max Depth'),
        min_samples_leaf=IntSlider(value=5, min=1, max=50, step=1, description='Min Leaf Samples'),
        min_samples_split=IntSlider(value=2, min=2, max=50, step=1, description='Min Split Samples'),
        max_leaf_nodes=IntSlider(value=0, min=0, max=30, step=1, description='Max Leaf Nodes')
    )
    def build_custom_tree(max_depth, min_samples_leaf, min_samples_split, max_leaf_nodes, show_plot=True):
        """Interactive decision tree builder that returns the trained tree"""

        # Handle 0 as "None" for max_leaf_nodes
        kwargs = {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "random_state": 42
        }
        if max_leaf_nodes > 0:
            kwargs["max_leaf_nodes"] = max_leaf_nodes

        # Fit tree
        custom_tree = DecisionTreeClassifier(**kwargs)
        custom_tree.fit(X, y)
        predictions = custom_tree.predict(X)
        accuracy = accuracy_score(y, predictions)
        importance = custom_tree.feature_importances_

        if not show_plot:
            return custom_tree

        # Setup figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Tree (Depth: {custom_tree.get_depth()}, Leaves: {custom_tree.get_n_leaves()}, Accuracy: {accuracy:.1%})")

        # --- Horizontal bar "gauge" for accuracy ---
        axs[0].barh(["Accuracy"], [accuracy * 100], color='green')
        axs[0].set_xlim(0, 100)
        axs[0].set_title("Accuracy Gauge")
        axs[0].set_xlabel("Percent")
        axs[0].grid(True, axis='x', linestyle='--', alpha=0.5)
        axs[0].axvspan(0, 60, color='#ff4d4d', alpha=0.2)
        axs[0].axvspan(60, 80, color='#ffa64d', alpha=0.2)
        axs[0].axvspan(80, 90, color='#d4f542', alpha=0.2)
        axs[0].axvspan(90, 100, color='#4dff88', alpha=0.2)

        # --- Feature importance bar plot ---
        axs[1].bar(X.columns, importance, color='teal')
        axs[1].set_title("Feature Importance")
        axs[1].set_ylabel("Importance")
        axs[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        if accuracy > 0.8:
            print("\n🎉 Excellent accuracy! But be careful of overfitting...")
        elif accuracy > 0.75:
            print("\n✅ Good performance!")
        else:
            print("\n⚠️ Try adjusting parameters for better performance")


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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(depths), y=train_accs, mode='lines+markers', name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=list(depths), y=test_accs, mode='lines+markers', name='Test Accuracy'))
    fig.update_layout(title="Effect of Tree Depth on Accuracy", xaxis_title="Max Depth", yaxis_title="Accuracy")
    fig.show()


def plot_model_comparison(results):
    """Plot comparison of different models' accuracies."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(results.keys()), y=list(results.values()),
                         text=[f"{v:.1%}" for v in results.values()],
                         textposition='auto', marker_color=["green", "blue", "orange"]))

    fig.update_layout(title="📊 Model Comparison on Pizza Preference",
                      yaxis_title="Accuracy", xaxis_title="Model",
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


def generate_pizza_data(n_samples):
    """Generate synthetic pizza preference dataset."""
    np.random.seed(42)  # for reproducibility

    # --- Core features ---
    ages = np.random.randint(5, 70, size=n_samples)
    likes_cheese = np.random.binomial(1, 0.75, size=n_samples)      # Most like cheese
    vegetarian = np.random.binomial(1, 0.3, size=n_samples)         # Minority vegetarian
    has_pet = np.random.binomial(1, 0.5, size=n_samples)

    # --- Extra features ---
    num_siblings = np.random.poisson(1.5, size=n_samples)           # Adds noise & pattern
    favorite_topping = np.random.choice(['pepperoni', 'mushroom', 'pineapple'], size=n_samples)

    # One-hot encode topping
    topping_dummies = pd.get_dummies(favorite_topping, prefix='topping')

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
    prob += np.random.normal(0, 0.05, size=n_samples)

    # Clip to valid probability range
    prob = np.clip(prob, 0, 1)

    # Final target variable
    likes_pizza = np.random.binomial(1, prob)

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

