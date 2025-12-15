
import matplotlib.pyplot as plt

def plot_model_results(y_test, y_pred, title="Model Results", save_path=None):
    """
    Plots the true values vs. predicted values for a regression model.
    
    Parameters:
        y_test (array-like): The true target values.
        y_pred (array-like): The predicted target values.
        title (str): Title of the plot.
    """
    # Ensure inputs are numpy arrays for indexing
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Sort values for a smoother line graph
    sorted_indices = np.argsort(y_test)
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(y_pred_sorted, label="Predicted Values", linestyle='-', marker='x', markersize=4)
    plt.plot(y_test_sorted, label="True Values", linestyle='-', marker='o', markersize=4)
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel("Samples (Sorted by True Values)", fontsize=14)
    plt.ylabel("traction", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Show the plot
    #plt.show()
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        #print(f"Plot saved to: {save_path}")