#Program 9
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(0) 
x = np.random.rand(100) 
y = 2 * x + np.random.normal(0, 0.1, 100)
def pearson_correlation(x, y): 
    n = len(x) 
    sum_x = np.sum(x) 
    sum_y = np.sum(y) 
    sum_x2 = np.sum(x**2) 
    sum_y2 = np.sum(y**2) 
    sum_xy = np.sum(x * y) 
    numerator = n * sum_xy - sum_x * sum_y 
    denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - 
sum_y**2)) 
    return numerator / denominator 
correlation = pearson_correlation(x, y) 
print(f"Pearson Correlation Coefficient: {correlation}") 
def spearman_rank_correlation(x, y): 
    rank_x = np.argsort(np.argsort(x)) 
    rank_y = np.argsort(np.argsort(y)) 
    return pearson_correlation(rank_x, rank_y) 
rank_correlation = spearman_rank_correlation(x, y) 
print(f"Spearman Rank Correlation Coefficient: {rank_correlation}") 
def linear_regression(x, y): 
    n = len(x) 
    m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * 
np.sum(x**2) - 
    (np.sum(x)**2)) 
    b = (np.sum(y) - m * np.sum(x)) / n 
    return m, b 
slope, intercept = linear_regression(x, y) 
print(f"Linear Regression: Slope = {slope}, Intercept = {intercept}") 
plt.scatter(x, y, label='Data Points') 
plt.plot(x, slope * x + intercept, color='red', label='Regression Line') 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.title('Scatter Plot with Regression Line') 
plt.legend() 
plt.savefig("outputs/09_regression.png", dpi=150, bbox_inches="tight")
plt.close()
def plot_correlation_matrix(x, y): 
    correlation_matrix = np.corrcoef(x, y) 
    plt.imshow(correlation_matrix, cmap='hot', 
interpolation='nearest') 
    plt.colorbar() 
    plt.title('Correlation Matrix Heat Map') 
    plt.xticks([0, 1], ['X', 'Y']) 
    plt.yticks([0, 1], ['X', 'Y']) 
    plt.savefig("outputs/09_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
plot_correlation_matrix(x, y)