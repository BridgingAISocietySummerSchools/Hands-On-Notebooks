"""
Visualization functions for ML fundamentals notebook.

This module contains plotting functions extracted from the notebook
to reduce code-heavy appearance while preserving core teaching content.
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, FloatSlider, FloatLogSlider, IntSlider


def plot_house_data_scatter(house_sizes, house_prices):
    """Plot house sizes vs prices as a scatter plot using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=house_sizes, y=house_prices,
        mode='markers',
        marker=dict(size=12, color='blue'),
        name='House Sales',
        hovertemplate='Size: %{x:,} sq ft<br>Price: $%{y}k<extra></extra>'
    ))

    fig.update_layout(
        title="🏠 House Prices vs Size - Can You See the Pattern?",
        xaxis_title="House Size (sq ft)",
        yaxis_title="Price ($1000s)",
        height=400,
        showlegend=False
    )

    fig.show()


def create_manual_line_interactive(house_sizes, house_prices):
    """Create interactive widget for manually adjusting regression line."""
    @interact(
        slope=FloatSlider(
            value=0.00,
            min=0.0,
            max=1.0,
            step=0.01,
            description='Slope'
        ),
        intercept=FloatSlider(
            value=300,
            min=-200,
            max=400,
            step=1,
            description='Intercept'
        )
    )
    def plot_manual_line(slope, intercept):
        """Interactive tool to manually adjust the line"""

        # Calculate predictions with manual line
        predictions = slope * house_sizes + intercept

        # Calculate error
        error = np.mean((house_prices - predictions) ** 2)

        # Create plot
        plt.figure(figsize=(8, 5))
        plt.scatter(house_sizes, house_prices, color='blue', label='Actual Prices', s=50)
        plt.plot(house_sizes, predictions, color='red', linewidth=2.5, label='Your Line')

        plt.title(f"Price = {slope:.3f} × Size + {intercept:.1f}   |   Error: {error:.1f}")
        plt.xlabel("House Size (sq ft)")
        plt.ylabel("Price ($1000s)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_computer_best_line(house_sizes, house_prices, best_slope, best_intercept, best_predictions, best_error):
    """Plot the computer's best regression line using Plotly."""
    fig = go.Figure()

    # Original data
    fig.add_trace(go.Scatter(
        x=house_sizes, y=house_prices,
        mode='markers',
        marker=dict(size=12, color='blue'),
        name='Actual Prices'
    ))

    # Computer's best line
    fig.add_trace(go.Scatter(
        x=house_sizes, y=best_predictions,
        mode='lines',
        line=dict(color='green', width=3),
        name='Computer\'s Best Line'
    ))

    fig.update_layout(
        title=f"🤖 Computer's Best Line: Price = ${best_slope:.3f} x Size + ${best_intercept:.1f} (Error: {best_error:.2f})",
        xaxis_title="House Size (sq ft)",
        yaxis_title="Price ($1000)",
        height=400
    )

    fig.show()


def plot_learning_process(history):
    """Visualize the gradient descent learning process."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{}, {"secondary_y": True}]],
        subplot_titles=('Error Decreases Over Time', 'Parameters Converge to Solution')
    )

    # Error over time
    fig.add_trace(
        go.Scatter(x=history['step'], y=history['error'],
                mode='lines+markers', name='Error',
                line=dict(color='red', width=2)),
        row=1, col=1
    )

    # Parameters over time
    fig.add_trace(
        go.Scatter(x=history['step'], y=history['slope'],
                mode='lines', name='Slope',
                line=dict(color='blue', width=2)),
        row=1, col=2, secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=history['step'], y=history['intercept'],
                mode='lines', name='Intercept',
                line=dict(color='green', width=2)),
        row=1, col=2, secondary_y=True
    )

    fig.update_layout(height=400, title_text="🏔️ Gradient Descent: Going Downhill to Find the Solution")
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_yaxes(title_text="Slope", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Intercept", row=1, col=2, secondary_y=True)

    fig.show()


def create_learning_rate_interactive(house_sizes_norm, house_prices, std_size, mean_size, mean_price, 
                                   denormalize_slope_func, denormalize_intercept_func):
    """Create interactive widget for testing different learning rates."""
    @interact(
        learning_rate=FloatLogSlider(
            value=0.01,
            base=10,
            min=-4,  # 0.0001
            max=0,   # 1.0
            step=0.1,
            description='Learning Rate'
        ),
        n_steps=IntSlider(
            value=70,
            min=1,
            max=200,
            step=1,
            description='Steps'
        )
    )
    def test_learning_rate(learning_rate, n_steps):
        slope, intercept = 0.0, 0.0
        errors = []

        for _ in range(n_steps):
            predictions = slope * house_sizes_norm + intercept
            error = np.mean((house_prices - predictions) ** 2)
            errors.append(error)

            # Calculate gradients
            n = len(house_sizes_norm)
            error_diff = house_prices - predictions
            slope_gradient = -2 * np.sum(error_diff * house_sizes_norm) / n
            intercept_gradient = -2 * np.sum(error_diff) / n

            # Update parameters
            slope = slope - learning_rate * slope_gradient
            intercept = intercept - learning_rate * intercept_gradient

        # Plot results
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(n_steps), errors, color='purple', linewidth=2.5, marker='o', markersize=4)
        ax.set_xlabel("Step")
        ax.set_ylabel("Error")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(0, max(1000, max(errors) * 1.1))
        plt.tight_layout()
        plt.show()

        final_slope = denormalize_slope_func(slope, std_size)
        final_intercept = denormalize_intercept_func(slope, mean_size, std_size, mean_price)

        print(f"Final: Slope={final_slope:.4f}, Intercept={final_intercept:.2f}, Error={errors[-1]:.2f}")

        if errors[-1] < 100:
            print("✅ Good performance!")
        elif learning_rate > 0.1:
            print("⚠️ Learning rate too high — diverging or unstable.")
        else:
            print("🐌 Learning rate too low — slow convergence.")


def plot_coffee_productivity(coffee_cups, tasks_done, coffee_model, cups_input, predicted_tasks):
    """Plot the coffee productivity example with regression line and prediction."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=coffee_cups, y=tasks_done, mode='markers',
                             marker=dict(size=10, color='brown'), name='Observed Data'))

    # Add prediction line
    pred_line = coffee_model.predict(coffee_cups.reshape(-1, 1))
    fig.add_trace(go.Scatter(x=coffee_cups, y=pred_line, mode='lines',
                             line=dict(color='darkred', width=3), name='Regression Line'))

    # Mark the prediction
    fig.add_trace(go.Scatter(x=[cups_input], y=[predicted_tasks], mode='markers',
                             marker=dict(size=15, color='crimson', symbol='star'),
                             name='Your Prediction'))

    fig.update_layout(title="☕ Coffee Intake vs Tasks Completed",
                      xaxis_title="Cups of Coffee per Day",
                      yaxis_title="Tasks Completed per Day",
                      height=400)
    fig.show()


# Utility functions for gradient descent
def normalize_data(data):
    """Normalize data for stable gradient descent"""
    return (data - np.mean(data)) / np.std(data), np.mean(data), np.std(data)


def denormalize_slope(norm_slope, std_x):
    """Convert normalized slope back to original scale"""
    return norm_slope / std_x


def denormalize_intercept(norm_slope, mean_x, std_x, mean_y):
    """Convert normalized intercept back to original scale"""
    orig_slope = denormalize_slope(norm_slope, std_x)
    return mean_y - orig_slope * mean_x

