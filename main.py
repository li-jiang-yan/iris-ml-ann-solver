from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load dataset
iris = datasets.load_iris()
A = iris.data
b = iris.target
num_classes = len(iris.target_names)
num_features = len(iris.feature_names)

# Scale data
A = StandardScaler().fit(A).transform(A)

# Split data into train and test data
A_train, A_test, b_train, b_test = train_test_split(A, b, random_state=1)

# Train model
x = Input(shape=(4,))
h = Dense(num_classes ** num_features, activation="sigmoid")(x)
y = Dense(num_classes, activation="sigmoid")(h)
model = Model(x, y)
optimizer = Adam(learning_rate=0.1)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(A_train, to_categorical(b_train, num_classes=num_classes), epochs=100)

# Classify test data using model
b_pred = np.argmax(model.predict(A_test), axis=1)

# Compute the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(b_test, b_pred), display_labels=iris.target_names)
disp.plot()
plt.show()

# Get the model classification metrics (will only show after the confusion matrix display window is closed)
print(classification_report(b_test, b_pred, target_names=iris.target_names))
