import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

breast = pd.read_csv('breastCancer.csv')
df = breast.iloc[:, 1:]
df['class'] = df['class'].apply(lambda x: "benign" if x == 2 else "malignant")
df.head()

encoder = LabelEncoder()
df['class'] = encoder.fit_transform(df['class'])
df.head()

df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'], errors='coerce')
print(df.dtypes)

X = df.drop('class', axis=1)
Y = df['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state=0)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_classifier = SVC()
svm_classifier.fit(X_train, Y_train)
y_pred = svm_classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]
cmap = sns.color_palette("Reds", as_cmap=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap=cmap, vmax=1.0, vmin=0.0)
plt.xticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])
plt.yticks(ticks=[0.5, 1.5], labels=["Bening", "Malignant"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix of SVM')
plt.show()
print("Confusion Matrix%:\n", cm_percent)

score = accuracy_score(y_pred, Y_test)
f1 = f1_score(Y_test, y_pred)
print("Accuracy:", score)
print("F1-score:", f1)
print(classification_report(Y_test, y_pred))

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
print("Sensitivity:", sensitivity)
specificity = tn / (tn + fp)
print("Specificity:", specificity)
precision = tp / (tp + fp)
print("Precision:", precision)
recall = tp / (tp + fn)
print("Recall:", recall)
auc_roc = roc_auc_score(Y_test, y_pred)
print("AUC-ROC:", auc_roc)

train_accuracy = svm_classifier.score(X_train, Y_train)
print("Train Accuracy:", train_accuracy)
cv_scores = cross_val_score(svm_classifier, X_train, Y_train)
print("Mean cross-validation score:", np.mean(cv_scores))
