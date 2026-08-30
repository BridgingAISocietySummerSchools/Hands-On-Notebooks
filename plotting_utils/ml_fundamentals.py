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
        title=(f"🤖 Computer's Best Line: Price = {best_slope:.3f} × Size "
               f"+ {best_intercept:.1f}  (MSE: {best_error:.2f})"),
        xaxis_title="House Size (sq ft)",
        yaxis_title="Price ($1000s)",
        height=400
    )

    fig.show()


def plot_learning_process(history, target_slope=None, target_intercept=None):
    """Visualize the gradient descent learning process.

    Pass target_slope / target_intercept to draw the optimal values as dashed
    reference lines, so the learning curves can be read against where they
    are heading.
    """
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

    # Dashed reference lines showing where each parameter should end up
    steps = history['step']
    if target_slope is not None:
        fig.add_trace(
            go.Scatter(x=[steps[0], steps[-1]], y=[target_slope, target_slope],
                       mode='lines', name='Slope (optimum)',
                       line=dict(color='blue', width=1.5, dash='dash')),
            row=1, col=2, secondary_y=False
        )

    if target_intercept is not None:
        fig.add_trace(
            go.Scatter(x=[steps[0], steps[-1]], y=[target_intercept, target_intercept],
                       mode='lines', name='Intercept (optimum)',
                       line=dict(color='green', width=1.5, dash='dash')),
            row=1, col=2, secondary_y=True
        )

    fig.update_layout(height=400, title_text="🏔️ Gradient Descent: Going Downhill to Find the Solution")
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_yaxes(title_text="Slope", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Intercept", row=1, col=2, secondary_y=True)

    fig.show()


def create_learning_rate_interactive(house_sizes_norm, house_prices, std_size, mean_size,
                                     denormalize_slope_func, denormalize_intercept_func):
    """Create interactive widget for testing different learning rates."""
    # Closed-form optimum, so the widget can say how close we actually got.
    # x is normalized (mean 0), so the best intercept is simply mean(y).
    opt_slope = np.sum(house_sizes_norm * house_prices) / np.sum(house_sizes_norm ** 2)
    opt_intercept = np.mean(house_prices)
    best_error = np.mean((house_prices - (opt_slope * house_sizes_norm + opt_intercept)) ** 2)

    @interact(
        learning_rate=FloatLogSlider(
            value=0.1,
            base=10,
            min=-4,   # 0.0001
            max=0.5,  # ~3.16 -- above 1.0 this problem genuinely diverges
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

        with np.errstate(over='ignore', invalid='ignore'):
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

        errors = np.array(errors, dtype=float)

        # Plot results. The error spans several orders of magnitude, so a linear
        # axis would squash everything interesting into the bottom pixel row.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(n_steps), errors, color='purple', linewidth=2.5, marker='o', markersize=4)
        ax.axhline(best_error, color='green', linestyle='--', linewidth=1.5,
                   label=f'Best possible error ({best_error:.1f})')
        if np.all(np.isfinite(errors)) and np.all(errors > 0):
            ax.set_yscale('log')
        ax.set_xlabel("Step")
        ax.set_ylabel("Error (log scale)")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.show()

        final = errors[-1]
        final_slope = denormalize_slope_func(slope, std_size)
        final_intercept = denormalize_intercept_func(slope, intercept, mean_size, std_size)
        print(f"Final: Slope={final_slope:.4f}, Intercept={final_intercept:.2f}, Error={final:.2f}")

        # 1% tolerance: at the critical rate the error is mathematically constant,
        # and float noise alone should not label that 'diverging'.
        if not np.isfinite(final) or final > errors[0] * 1.01:
            print("💥 Diverging — each step overshoots further than the last. Much too high.")
        elif final <= best_error * 1.05:
            print(f"✅ Converged — that is essentially the best possible error ({best_error:.2f}).")
        elif final >= errors[0] * 0.99:
            print("🌀 Stuck — the steps are so large the model just bounces between "
                  "two points and never settles. Slightly too high.")
        elif np.any(np.diff(errors) > 1e-9):
            print("⚠️ Overshooting — the error jumps back up along the way, but still "
                  "makes progress overall.")
        else:
            print(f"🐌 Too slow — still at {final:.2f} after {n_steps} steps "
                  f"(best possible: {best_error:.2f}). Raise the rate or add steps.")


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


def denormalize_intercept(norm_slope, norm_intercept, mean_x, std_x):
    """Convert the learned intercept back to the original scale.

    In normalized space the model is
        y = norm_slope * (x - mean_x) / std_x + norm_intercept
    which, expanded in the original units of x, is
        y = (norm_slope / std_x) * x + (norm_intercept - norm_slope * mean_x / std_x)

    Note this needs the *learned* norm_intercept. Substituting mean(y) for it
    would silently assume the intercept has already converged, and the returned
    value would then track the slope instead of the parameter it names.
    """
    orig_slope = denormalize_slope(norm_slope, std_x)
    return norm_intercept - orig_slope * mean_x

