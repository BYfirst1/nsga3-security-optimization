import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates, scatter_matrix
import pandas as pd

# Функция проверки доминирования
def dominates(p1, p2):
    return (p1[0] <= p2[0] and p1[1] >= p2[1] and p1[2] <= p2[2]) and \
           (p1[0] < p2[0] or p1[1] > p2[1] or p1[2] < p2[2])

# Функция поиска нескольких уровней фронтов Парето
def fast_non_dominated_sort(points):
    fronts = []
    num_points = len(points)
    domination_count = np.zeros(num_points, dtype=int)
    dominated_solutions = [[] for _ in range(num_points)]

    for i in range(num_points):
        for j in range(num_points):
            if dominates(points[i], points[j]):
                dominated_solutions[i].append(j)
            elif dominates(points[j], points[i]):
                domination_count[i] += 1

    front = np.where(domination_count == 0)[0].tolist()
    fronts.append(front)

    while front:
        next_front = []
        for i in front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        front = next_front

    return fronts

# Функция расчета гиперобъема
def hypervolume_monte_carlo(front, reference_point, samples=100000):
    random_points = np.random.uniform(low=front.min(axis=0), high=reference_point, size=(samples, 3))
    count_dominated = sum(any((p <= sol).all() for sol in front) for p in random_points)
    volume = np.prod(reference_point - front.min(axis=0)) * (count_dominated / samples)
    return volume

# Генерация данных
np.random.seed(42)
num_points = 200
cost = np.random.uniform(2000, 15000, num_points)
performance = np.random.uniform(0, 100, num_points)
risk = np.random.uniform(0, 1, num_points)

solutions = np.column_stack((cost, performance, risk))
pareto_fronts = fast_non_dominated_sort(solutions)
pareto_solutions = solutions[pareto_fronts[0]]

reference_point = np.max(solutions, axis=0) + [1000, -10, 0.1]
hv_values = []
for i in range(1, len(pareto_fronts) + 1):
    hv_values.append(hypervolume_monte_carlo(solutions[pareto_fronts[0][:i]], reference_point))

# 3D График Фронта Парето
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(solutions[:, 0], solutions[:, 1], solutions[:, 2], c='b', marker='o', alpha=0.3, label='All Solutions')
ax.scatter(pareto_solutions[:, 0], pareto_solutions[:, 1], pareto_solutions[:, 2], c='r', marker='^', s=100, label='Pareto Front')
ax.set_xlabel('Cost (min)')
ax.set_ylabel('Performance (max)')
ax.set_zlabel('Residual Risk (min)')
ax.set_title('3D Pareto Front Visualization')
ax.legend()
plt.show()

# Эволюция гиперобъема
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(hv_values) + 1), hv_values, marker='o', linestyle='-', color='purple')
plt.xlabel('Iteration')
plt.ylabel('Hypervolume')
plt.title('Hypervolume Evolution')
plt.show()

# Гистограмма плотности решений
plt.figure(figsize=(8, 6))
sns.histplot(solutions[:, 0], kde=True, color='blue', label='Cost')
sns.histplot(solutions[:, 1], kde=True, color='green', label='Performance')
sns.histplot(solutions[:, 2], kde=True, color='red', label='Residual Risk')
plt.legend()
plt.title('Density Histogram of Solutions')
plt.show()

# График распределения доминируемых решений
dominance_counts = [sum(dominates(sol, other) for other in solutions) for sol in solutions]
plt.figure(figsize=(8, 6))
plt.hist(dominance_counts, bins=30, color='cyan', edgecolor='black')
plt.xlabel('Number of Dominated Solutions')
plt.ylabel('Frequency')
plt.title('Distribution of Dominated Solutions')
plt.show()

# Scatter Plot Matrix
df = pd.DataFrame(solutions, columns=['Cost', 'Performance', 'Residual Risk'])
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha':0.5})
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()

# График сходимости Pareto Front
plt.figure(figsize=(8, 6))
iterations = np.arange(1, len(pareto_fronts) + 1)
plt.plot(iterations, [len(front) for front in pareto_fronts], marker='s', linestyle='-', color='darkorange')
plt.xlabel('Iteration')
plt.ylabel('Number of Pareto Solutions')
plt.title('Pareto Front Convergence')
plt.show()
