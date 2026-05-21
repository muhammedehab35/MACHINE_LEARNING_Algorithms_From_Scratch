"""
Generate all visualization images for Naive Bayes README
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Create images directory
os.makedirs('images', exist_ok=True)

print("Generating Naive Bayes visualizations...")
print("=" * 70)

# 1. Gaussian NB Decision Boundaries
print("\n1. Generating Gaussian NB decision boundaries...")
iris = load_iris()
X = iris.data[:, :2]  # Use only first 2 features for visualization
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Create mesh
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Get probabilities for background
proba = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
max_proba = np.max(proba, axis=1).reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Decision boundaries
ax1.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis',
                      edgecolors='black', s=100, alpha=0.8, label='Train')
ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis',
            edgecolors='red', s=100, linewidths=2, alpha=0.8, marker='s', label='Test')
ax1.set_xlabel('Sepal Length (cm)', fontsize=12)
ax1.set_ylabel('Sepal Width (cm)', fontsize=12)
ax1.set_title('Gaussian NB Decision Boundaries\nIris Dataset', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Posterior probability heatmap
im = ax2.contourf(xx, yy, max_proba, levels=20, cmap='RdYlGn', alpha=0.8)
ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis',
           edgecolors='black', s=50, alpha=0.6)
ax2.set_xlabel('Sepal Length (cm)', fontsize=12)
ax2.set_ylabel('Sepal Width (cm)', fontsize=12)
ax2.set_title('Posterior Probability Confidence', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Max Posterior Probability', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/gaussian_decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/gaussian_decision_boundaries.png")

# 2. Multinomial NB Text Classification
print("\n2. Generating Multinomial NB text classification...")
np.random.seed(42)

# Simulate word count data
n_samples = 200
n_features = 10
X_text = np.random.randint(0, 10, (n_samples, n_features))
y_text = (X_text[:, 0] + X_text[:, 1] > 8).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_text, y_text, test_size=0.3, random_state=42)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train, y_train)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Word frequency per class
word_names = [f'Word {i+1}' for i in range(n_features)]
class_0_counts = X_train[y_train == 0].mean(axis=0)
class_1_counts = X_train[y_train == 1].mean(axis=0)

x_pos = np.arange(n_features)
width = 0.35

ax1.bar(x_pos - width/2, class_0_counts, width, label='Class 0 (Negative)', alpha=0.8, color='coral')
ax1.bar(x_pos + width/2, class_1_counts, width, label='Class 1 (Positive)', alpha=0.8, color='lightblue')
ax1.set_xlabel('Word Index', fontsize=12)
ax1.set_ylabel('Average Count', fontsize=12)
ax1.set_title('Word Frequency Distribution per Class', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(word_names, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Feature importance
feature_log_prob_diff = np.abs(mnb.feature_log_prob_[1] - mnb.feature_log_prob_[0])
sorted_indices = np.argsort(feature_log_prob_diff)[::-1]

ax2.barh(word_names, feature_log_prob_diff, color='steelblue', alpha=0.8)
ax2.set_xlabel('Log Probability Difference', fontsize=12)
ax2.set_ylabel('Words', fontsize=12)
ax2.set_title('Feature Importance (Top Words)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Confusion matrix
y_pred = mnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

im3 = ax3.imshow(cm, cmap='Blues', interpolation='nearest')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Negative', 'Positive'])
ax3.set_yticklabels(['Negative', 'Positive'])
ax3.set_xlabel('Predicted Label', fontsize=12)
ax3.set_ylabel('True Label', fontsize=12)
ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontsize=16, fontweight='bold')

cbar3 = plt.colorbar(im3, ax=ax3)

# Accuracy comparison
train_acc = mnb.score(X_train, y_train)
test_acc = mnb.score(X_test, y_test)

metrics = ['Training Accuracy', 'Test Accuracy']
values = [train_acc * 100, test_acc * 100]
colors = ['lightgreen', 'lightblue']

ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.set_title(f'MultinomialNB Performance\nTest Accuracy: {test_acc:.2%}',
             fontsize=14, fontweight='bold')
ax4.set_ylim([0, 105])
ax4.grid(True, alpha=0.3, axis='y')

for i, (metric, value) in enumerate(zip(metrics, values)):
    ax4.text(i, value + 1, f'{value:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('images/multinomial_text_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/multinomial_text_classification.png")

# 3. Bernoulli NB Binary Features
print("\n3. Generating Bernoulli NB binary features...")
np.random.seed(42)

# Generate binary data
X_binary, y_binary = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                        n_informative=2, n_clusters_per_class=1,
                                        random_state=42, flip_y=0.1)
X_binary = (X_binary > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Binary feature patterns
patterns = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
pattern_names = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']

class_0_patterns = []
class_1_patterns = []

for pattern in patterns:
    mask = np.all(X_train == pattern, axis=1)
    class_0_patterns.append(np.sum(mask & (y_train == 0)))
    class_1_patterns.append(np.sum(mask & (y_train == 1)))

x_pos = np.arange(len(patterns))
width = 0.35

ax1.bar(x_pos - width/2, class_0_patterns, width, label='Class 0', alpha=0.8, color='coral')
ax1.bar(x_pos + width/2, class_1_patterns, width, label='Class 1', alpha=0.8, color='lightblue')
ax1.set_xlabel('Binary Pattern', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Binary Feature Presence/Absence Patterns', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(pattern_names)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Decision boundaries
x_grid, y_grid = np.meshgrid([0, 1], [0, 1])
grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]
predictions = bnb.predict(grid_points).reshape(x_grid.shape)

im2 = ax2.imshow(predictions, cmap='RdYlBu', alpha=0.3, extent=[-0.5, 1.5, -0.5, 1.5])
ax2.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='red',
           marker='o', s=100, alpha=0.6, label='Class 0', edgecolors='black')
ax2.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='blue',
           marker='s', s=100, alpha=0.6, label='Class 1', edgecolors='black')
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.set_title('Decision Boundaries (Binary Features)', fontsize=14, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Feature importance
feature_names = ['Feature 1', 'Feature 2']
log_prob_diff = np.abs(bnb.feature_log_prob_[1] - bnb.feature_log_prob_[0])

ax3.bar(feature_names, log_prob_diff, color='steelblue', alpha=0.8)
ax3.set_ylabel('Log Probability Difference', fontsize=12)
ax3.set_title('Feature Importance for Sentiment', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Performance metrics
train_acc = bnb.score(X_train, y_train)
test_acc = bnb.score(X_test, y_test)

metrics = ['Training', 'Test']
values = [train_acc * 100, test_acc * 100]

ax4.bar(metrics, values, color=['lightgreen', 'lightblue'], alpha=0.8, edgecolor='black')
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.set_title(f'BernoulliNB Accuracy\nTest: {test_acc:.2%}', fontsize=14, fontweight='bold')
ax4.set_ylim([0, 105])
ax4.grid(True, alpha=0.3, axis='y')

for i, value in enumerate(values):
    ax4.text(i, value + 1, f'{value:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('images/bernoulli_binary_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/bernoulli_binary_features.png")

# 4. Variants Comparison
print("\n4. Generating variants comparison...")
np.random.seed(42)

# Test on different data types
results = {'GaussianNB': [], 'MultinomialNB': [], 'BernoulliNB': []}
data_types = ['Continuous', 'Count', 'Binary']

# Continuous data
X_cont, y_cont = make_classification(n_samples=300, n_features=10, n_informative=8,
                                     n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_cont, y_cont, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
results['GaussianNB'].append(gnb.score(X_test, y_test) * 100)

X_train_pos = X_train - X_train.min() + 1
X_test_pos = X_test - X_test.min() + 1
mnb = MultinomialNB()
mnb.fit(X_train_pos, y_train)
results['MultinomialNB'].append(mnb.score(X_test_pos, y_test) * 100)

X_train_bin = (X_train > 0).astype(int)
X_test_bin = (X_test > 0).astype(int)
bnb = BernoulliNB()
bnb.fit(X_train_bin, y_train)
results['BernoulliNB'].append(bnb.score(X_test_bin, y_test) * 100)

# Count data
X_count = np.random.poisson(3, (300, 10))
y_count = (X_count[:, 0] + X_count[:, 1] > 6).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_count, y_count, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
results['GaussianNB'].append(gnb.score(X_test, y_test) * 100)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
results['MultinomialNB'].append(mnb.score(X_test, y_test) * 100)

bnb = BernoulliNB()
bnb.fit((X_train > 2).astype(int), y_train)
results['BernoulliNB'].append(bnb.score((X_test > 2).astype(int), y_test) * 100)

# Binary data
X_bin, y_bin = make_classification(n_samples=300, n_features=10, n_informative=8,
                                   n_redundant=2, random_state=42)
X_bin = (X_bin > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
results['GaussianNB'].append(gnb.score(X_test, y_test) * 100)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
results['MultinomialNB'].append(mnb.score(X_test, y_test) * 100)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
results['BernoulliNB'].append(bnb.score(X_test, y_test) * 100)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Accuracy comparison
x = np.arange(len(data_types))
width = 0.25

ax1.bar(x - width, results['GaussianNB'], width, label='GaussianNB', alpha=0.8, color='coral')
ax1.bar(x, results['MultinomialNB'], width, label='MultinomialNB', alpha=0.8, color='lightblue')
ax1.bar(x + width, results['BernoulliNB'], width, label='BernoulliNB', alpha=0.8, color='lightgreen')

ax1.set_xlabel('Data Type', fontsize=12)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy Across Different Data Types', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(data_types)
ax1.legend()
ax1.set_ylim([0, 105])
ax1.grid(True, alpha=0.3, axis='y')

# Speed comparison (simulated)
import time

speeds = {'GaussianNB': [], 'MultinomialNB': [], 'BernoulliNB': []}
X_speed, y_speed = make_classification(n_samples=1000, n_features=20, random_state=42)

for _ in range(5):
    start = time.time()
    gnb = GaussianNB()
    gnb.fit(X_speed, y_speed)
    speeds['GaussianNB'].append(time.time() - start)

    X_speed_pos = X_speed - X_speed.min() + 1
    start = time.time()
    mnb = MultinomialNB()
    mnb.fit(X_speed_pos, y_speed)
    speeds['MultinomialNB'].append(time.time() - start)

    start = time.time()
    bnb = BernoulliNB()
    bnb.fit((X_speed > 0).astype(int), y_speed)
    speeds['BernoulliNB'].append(time.time() - start)

avg_speeds = [np.mean(speeds['GaussianNB']) * 1000,
              np.mean(speeds['MultinomialNB']) * 1000,
              np.mean(speeds['BernoulliNB']) * 1000]

variants = ['Gaussian', 'Multinomial', 'Bernoulli']
colors = ['coral', 'lightblue', 'lightgreen']

ax2.bar(variants, avg_speeds, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Training Time (ms)', fontsize=12)
ax2.set_title('Speed Comparison (1000 samples, 20 features)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

for i, speed in enumerate(avg_speeds):
    ax2.text(i, speed + max(avg_speeds)*0.02, f'{speed:.2f}ms',
            ha='center', fontsize=11, fontweight='bold')

# Best use cases
use_cases = {
    'Continuous\nFeatures': [1, 0, 0],
    'Count/Text\nData': [0, 1, 0],
    'Binary\nFeatures': [0, 0, 1]
}

x_pos = np.arange(len(use_cases))
for i, variant in enumerate(['GaussianNB', 'MultinomialNB', 'BernoulliNB']):
    values = [use_cases[uc][i] for uc in use_cases.keys()]
    ax3.bar(x_pos + i*width, values, width, label=variant,
           color=colors[i], alpha=0.8)

ax3.set_xlabel('Use Case', fontsize=12)
ax3.set_ylabel('Suitability', fontsize=12)
ax3.set_title('Best Use Case for Each Variant', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels(use_cases.keys())
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Not Suitable', 'Best Choice'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Summary table
summary_data = [
    ['Gaussian', f'{results["GaussianNB"][0]:.1f}%', f'{avg_speeds[0]:.2f}ms', 'Continuous'],
    ['Multinomial', f'{results["MultinomialNB"][1]:.1f}%', f'{avg_speeds[1]:.2f}ms', 'Count/Text'],
    ['Bernoulli', f'{results["BernoulliNB"][2]:.1f}%', f'{avg_speeds[2]:.2f}ms', 'Binary']
]

ax4.axis('tight')
ax4.axis('off')

table = ax4.table(cellText=summary_data,
                 colLabels=['Variant', 'Best Accuracy', 'Speed', 'Best For'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(4):
    table[(0, i)].set_facecolor('lightgray')
    table[(0, i)].set_text_props(weight='bold')

for i in range(1, 4):
    for j in range(4):
        table[(i, j)].set_facecolor(colors[i-1] if j == 0 else 'white')
        table[(i, j)].set_alpha(0.3)

ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('images/variants_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/variants_comparison.png")

# 5. Spam Detection Results
print("\n5. Generating spam detection results...")
np.random.seed(42)

# Simulate spam detection data
n_samples = 500
n_features = 20

# Create features that simulate email characteristics
X_spam = np.random.poisson(8, (n_samples//2, n_features))  # Spam emails
X_ham = np.random.poisson(3, (n_samples//2, n_features))   # Ham emails

X_all = np.vstack([X_spam, X_ham])
y_all = np.array([1] * (n_samples//2) + [0] * (n_samples//2))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)
y_proba = mnb.predict_proba(X_test)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

im1 = ax1.imshow(cm, cmap='Blues', interpolation='nearest')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Ham', 'Spam'])
ax1.set_yticklabels(['Ham', 'Spam'])
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_title('Confusion Matrix - Spam Detection', fontsize=14, fontweight='bold')

for i in range(2):
    for j in range(2):
        text = ax1.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontsize=20, fontweight='bold')

cbar1 = plt.colorbar(im1, ax=ax1)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

ax2.plot(fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax2.fill_between(fpr, tpr, alpha=0.2, color='orange')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# Top spam indicator words
feature_names = [f'Word{i+1}' for i in range(n_features)]
spam_indicators = mnb.feature_log_prob_[1] - mnb.feature_log_prob_[0]
top_spam_indices = np.argsort(spam_indicators)[-10:][::-1]

ax3.barh(range(10), spam_indicators[top_spam_indices], color='red', alpha=0.7)
ax3.set_yticks(range(10))
ax3.set_yticklabels([feature_names[i] for i in top_spam_indices])
ax3.set_xlabel('Log Probability Ratio (Spam/Ham)', fontsize=12)
ax3.set_title('Top 10 Spam Indicator Words', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Performance metrics
accuracy = np.sum(y_pred == y_test) / len(y_test)
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
colors_metrics = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral']

bars = ax4.bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black')
ax4.set_ylabel('Score (%)', fontsize=12)
ax4.set_title(f'Spam Detection Performance\nAccuracy: {accuracy:.2%}',
             fontsize=14, fontweight='bold')
ax4.set_ylim([0, 105])
ax4.grid(True, alpha=0.3, axis='y')

for i, (metric, value) in enumerate(zip(metrics, values)):
    ax4.text(i, value + 1, f'{value:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('images/spam_detection_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: images/spam_detection_results.png")

print("\n" + "=" * 70)
print("All Naive Bayes visualizations generated successfully!")
print("Images saved in: 08_NAIVE_BAYES/images/")
print("=" * 70)
