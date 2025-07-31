"""
Neural Networks visualization functions for ML fundamentals notebook.

This module contains plotting functions extracted from the neural networks notebook
to reduce code-heavy appearance while preserving core teaching content.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ipywidgets import interact, FloatSlider, IntSlider


def plot_customer_data_scatter(customer_data):
    """Plot customer purchase patterns as a scatter plot."""
    fig = px.scatter(customer_data, x='age', y='income', color='will_buy',
                    color_discrete_map={0: 'red', 1: 'green'},
                    title="🛒 Customer Purchase Patterns",
                    labels={'will_buy': 'Will Buy'})
    fig.show()


def simple_neuron(age, income, weight_age=0.1, weight_income=0.05, bias=-3):
    """Simulate a simple neuron with sigmoid activation."""
    weighted_sum = weight_age * age + weight_income * income + bias

    # Let's use a sigmoid activation function
    activation = 1 / (1 + np.exp(-weighted_sum))
    prediction = 1 if activation > 0.5 else 0

    return prediction, activation


def create_interactive_neuron_designer(customer_data):
    """Create interactive widget for designing a single neuron."""
    @interact(
        weight_age=FloatSlider(value=0.1, min=-0.5, max=0.5, step=0.01, description='Weight (Age)'),
        weight_income=FloatSlider(value=0.05, min=-0.5, max=0.5, step=0.01, description='Weight (Income)'),
        bias=FloatSlider(value=-3.0, min=-10, max=10, step=0.1, description='Bias')
    )
    def design_neuron(weight_age, weight_income, bias):
        """Interactive tool to visualize a single neuron's decision boundary"""
        predictions = []
        confidences = []

        for _, row in customer_data.iterrows():
            pred, conf = simple_neuron(row['age'], row['income'], weight_age, weight_income, bias)
            predictions.append(pred)
            confidences.append(conf)

        accuracy = np.mean(np.array(predictions) == customer_data['will_buy'].values)

        # Right plot: decision boundary via contour
        age_range = np.linspace(20, 60, 50)
        income_range = np.linspace(20, 100, 50)
        Age, Income = np.meshgrid(age_range, income_range)

        Z = np.zeros_like(Age)
        for i in range(Age.shape[0]):
            for j in range(Age.shape[1]):
                _, conf = simple_neuron(Age[i, j], Income[i, j], weight_age, weight_income, bias)
                Z[i, j] = conf

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Neuron Accuracy: {accuracy:.1%}")

        # Left: prediction colors
        color_map = ['red' if pred != true else 'green'
                     for pred, true in zip(predictions, customer_data['will_buy'])]
        axes[0].scatter(customer_data['age'], customer_data['income'], c=color_map, s=50, edgecolors='k')
        axes[0].set_title("Prediction Accuracy")
        axes[0].set_xlabel("Age")
        axes[0].set_ylabel("Income ($k)")
        axes[0].grid(True)

        # Right: decision boundary with contour
        contour = axes[1].contourf(Age, Income, Z, levels=np.linspace(0, 1, 11), cmap='RdYlGn', alpha=0.7)
        scatter = axes[1].scatter(customer_data['age'], customer_data['income'],
                                  c=customer_data['will_buy'], cmap='RdYlGn', edgecolors='black')
        axes[1].set_title("Decision Boundary")
        axes[1].set_xlabel("Age")
        axes[1].set_ylabel("Income ($k)")
        axes[1].grid(True)

        fig.colorbar(contour, ax=axes[1], label="Confidence")

        plt.tight_layout()
        plt.show()

        # Console summary
        print(f"🧠 Your Neuron Performance:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Weights: Age={weight_age:.2f}, Income={weight_income:.3f}")
        print(f"   Bias: {bias:.2f}")

        if accuracy >= 0.8:
            print("\n🎉 Excellent! Your neuron learned the pattern well!")
        elif accuracy >= 0.6:
            print("\n✅ Good! Try adjusting weights for better performance.")
        else:
            print("\n🤔 Keep experimenting with the weights and bias!")


def plot_circle_data_scatter(circle_data):
    """Plot the complex circular pattern dataset."""
    fig = px.scatter(circle_data, x='x', y='y', color='class',
                    color_discrete_map={0: 'red', 1: 'blue'},
                    title="🎯 Complex Pattern: Circles within Circles")

    fig.update_layout(height=400)
    fig.show()


def plot_neural_network_results(X, y, model, scaler, history, test_accuracy):
    """Plot neural network decision boundary and training curves."""
    # Create meshgrid for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict over the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    Z = model.predict(grid_points_scaled, verbose=0)
    Z = Z.reshape(xx.shape)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Neural Network Decision Boundary', 'Training Loss Curve'),
        specs=[[{"type": "contour"}, {"type": "xy"}]]
    )

    # Subplot 1: Decision boundary
    fig.add_trace(
        go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale='RdYlBu',
            showscale=False,
            contours=dict(start=0, end=1, size=0.05),
            opacity=0.6,
        ),
        row=1, col=1
    )

    point_colors = ['red' if label == 0 else 'blue' for label in y]

    fig.add_trace(
        go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(
                color=point_colors,
                line=dict(width=1, color='black'),
                size=8
            ),
            name='Data Points'
        ),
        row=1, col=1
    )

    # Subplot 2: Training and validation loss
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history.history['loss']) + 1)),
            y=history.history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(width=3, color='green')
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history.history['val_loss']) + 1)),
            y=history.history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(width=3, dash='dash', color='blue')
        ),
        row=1, col=2
    )

    # Layout
    fig.update_layout(
        height=500,
        title_text=f"🧠 Neural Network: Decision Boundary and Loss Curve (Test Accuracy: {test_accuracy:.1%})",
        showlegend=True
    )

    fig.update_xaxes(title_text="X1", row=1, col=1)
    fig.update_yaxes(title_text="X2", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=2)

    fig.show()


def plot_digit_samples(X_digits, y_digits):
    """Display sample handwritten digits."""
    fig = make_subplots(
        rows=2, cols=5,
        subplot_titles=[f'Digit {i}' for i in range(10)]
    )

    for i in range(10):
        idx = (y_digits == i).nonzero()[0][0]
        image = X_digits[idx].reshape(8, 8)

        row = 1 if i < 5 else 2
        col = (i % 5) + 1

        fig.add_trace(
            go.Heatmap(z=image, colorscale='gray', showscale=False),
            row=row, col=col
        )

    fig.update_layout(height=400, title_text="🖼️ Sample Handwritten Digits (8x8 pixels)")
    fig.show()


def create_interactive_digit_classifier(X_test_digits, y_test_digits, X_test_digits_scaled, model_digits):
    """Create interactive widget for testing digit classification."""
    @interact(
        digit_index=IntSlider(
            value=0,
            min=0,
            max=len(X_test_digits) - 1,
            step=1,
            description='Digit #:',
            continuous_update=False
        )
    )
    def test_digit_classifier_keras(digit_index):
        """Interactive digit classification tool using Keras model"""

        # Safeguard against out-of-bounds index
        if digit_index >= len(X_test_digits):
            digit_index = len(X_test_digits) - 1

        test_image = X_test_digits[digit_index]
        true_label = y_test_digits[digit_index]

        # Predict with Keras model
        input_scaled = X_test_digits_scaled[digit_index].reshape(1, -1)
        probabilities = model_digits.predict(input_scaled, verbose=0)[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Left: Show the digit image
        axs[0].imshow(test_image.reshape(8, 8), cmap='gray')
        axs[0].set_title("Test Image")
        axs[0].axis('off')

        # Right: Bar chart of predicted probabilities
        bars = axs[1].bar(range(10), probabilities, color='lightblue')
        bars[prediction].set_color('green')
        axs[1].axvline(true_label, color='red', linestyle='--')
        axs[1].text(true_label, 1.05, f"True: {true_label}", color='red', ha='center')
        axs[1].set_title("Network Prediction")
        axs[1].set_xlabel("Digit")
        axs[1].set_ylabel("Probability")
        axs[1].set_xticks(range(10))
        axs[1].set_ylim(0, 1.1)

        plt.tight_layout()
        plt.show()

        # Prepare summary
        result = "✅ CORRECT" if prediction == true_label else "❌ WRONG"
        print(f"🎯 Classification Summary:")
        print(f"   True Digit: {true_label}")
        print(f"   Predicted : {prediction}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Result    : {result}")

        top_3 = np.argsort(probabilities)[-3:][::-1]
        print(f"\n🏆 Top 3 Predictions:")
        for rank, idx in enumerate(top_3, start=1):
            print(f"   {rank}. Digit {idx}: {probabilities[idx]:.1%}")
