import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv('exampledataset2.csv')

# Select the columns for sentiment analysis
text_column = 'text'
sentiment_column = 'sentiment'

X = df[text_column].to_numpy()
y = df[sentiment_column].to_numpy()

# With only 50 labeled rows, a single 80/20 train/test split evaluates on just 10 examples and
# swings wildly depending on which rows land in the test set. Stratified 5-fold cross-validation
# uses every row for both training and testing across folds, giving a mean +/- std estimate
# instead of one potentially lucky (or unlucky) accuracy number.
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit the vectorizer on the training fold only, so test-fold vocabulary never leaks in.
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = SVC()
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)

    fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    fold_metrics['precision'].append(precision_score(y_test, y_pred, pos_label='positive', zero_division=0))
    fold_metrics['recall'].append(recall_score(y_test, y_pred, pos_label='positive', zero_division=0))
    fold_metrics['f1'].append(f1_score(y_test, y_pred, pos_label='positive', zero_division=0))

    print(f'Fold {fold_index}: accuracy={fold_metrics["accuracy"][-1]:.2f}')

print(f'\n{n_splits}-fold cross-validation results (n={len(df)} labeled examples):')
for metric_name, values in fold_metrics.items():
    print(f'  {metric_name}: {np.mean(values):.3f} +/- {np.std(values):.3f}')

print(
    '\nNote: with only 50 labeled examples, these estimates still carry wide uncertainty.'
    ' Treat this as a baseline, not a validated production accuracy figure -- grow the'
    ' labeled dataset before reporting a single headline number.'
)

