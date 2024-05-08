import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0)

accuracy = model.evaluate(X_test, Y_test)
print('Accuracy:', accuracy)

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate other metrics
cm = confusion_matrix(Y_test, y_pred)
cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]
cmap = sns.color_palette("Reds", as_cmap=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap=cmap, vmax=1.0, vmin=0.0)
plt.xticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])
plt.yticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix of ANN')
plt.show()
print("Confusion Matrix%:\n", cm_percent)

score = accuracy_score(Y_test, y_pred)
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
