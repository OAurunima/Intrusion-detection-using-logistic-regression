from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

print("----------Downloading the dataset----------")
# fetch dataset
rt_iot2022 = fetch_ucirepo(id=942)
# data (as pandas dataframes)
X = rt_iot2022.data.features
y = rt_iot2022.data.targets
df = pd.DataFrame(data=X)
# The next lines of code are only informative/diagnostic
print(rt_iot2022.metadata) # this allows you to see the dataset metadata
print(rt_iot2022.variables) # allows you to see the variables
print(df.head())

#Initialize the LabelEncoder
label_encoder = LabelEncoder()

#Fit and transform the specified column using label encoding
X['encoded_services'] = label_encoder.fit_transform(X['service'])

#Print encoded column
print(X[['service', 'encoded_services']].head())

#Drop the column
X.drop('service', axis=1, inplace=True)

#Fit and transform the proto column using label encoding
X['encoded_proto'] = label_encoder.fit_transform(X['proto'])

##Print encoded column
print(X[['proto', 'encoded_proto']].head())

#Drop the column
X.drop('proto', axis=1, inplace=True)

unique_attack_types = y['Attack_type'].unique()

# Print unique var
print(unique_attack_types)

# Map the attack patterns to 1 and normal patterns to 0
attack_mapping = {'DOS_SYN_Hping': 1, 'ARP_poisioning': 1, 'NMAP_UDP_SCAN': 1, 'NMAP_XMAS_TREE_SCAN': 1, 'NMAP_OS_DETECTION': 1, 'NMAP_TCP_scan': 1, 'DDOS_Slowloris': 1, 'Metasploit_Brute_Force_SSH': 1,'NMAP_FIN_SCAN': 1, 'MQTT_Publish': 0, 'MQTT': 0, 'Thing_Speak': 0, 'Wipro_bulb': 0, 'Amazon-Alexa':0}

# Usemap to replace values in "Attack_type" column
y['Attack_type'] = y['Attack_type'].map(attack_mapping)

# Print the updated y
print(y)

unique_attack_types = y['Attack_type'].unique()

# Print unique var
print(unique_attack_types)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


hist_columns = [
    'bwd_header_size_max',
    'flow_FIN_flag_count',
    'payload_bytes_per_second'
]

for column in hist_columns:
    unique_values = X[column].unique()
    unique_values.sort()
    plt.hist(X[column], bins=20 if len(unique_values) > 20 else unique_values, label=column)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Histogram of ' + column)
    plt.show()


"""**No regularization**"""
print("""---------------No regularization----------------""")
# Initialize the logistic regression model with no regularization
model = LogisticRegression(penalty=None, tol = 0.001, solver = 'lbfgs')

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Compute confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

"""**L2 regularization (ridge)**"""
print("""---------------L2 regularization (ridge)----------------""")

# Initialize the logistic regression model with L2 regularization
model = LogisticRegression(penalty='l2', tol = 0.001, solver = 'lbfgs' )

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")


# Compute confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

"""**L1 regularization (Lasso)**"""
print("""---------------L1 regularization (Lasso)----------------""")


# Initialize the logistic regression model with L1 regularization
model = LogisticRegression(penalty='l1', tol = 0.01, solver = 'liblinear')

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Compute confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()



""" **L1-L2 regularization (elastic-net)**"""
print("""---------------L1-L2 regularization (elastic-net)----------------""")

# Initialize the logistic regression model with L1-L2 regularization (elastic-net) regularization
model = LogisticRegression(penalty='elasticnet', l1_ratio=0.551, tol = 0.001, solver= 'saga')

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Compute confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

"""**PCA dimensionality reduction**"""
print("""---------------PCA dimensionality reduction----------------""")

pca = PCA(n_components= 42)

X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

print(X_train_reduced.shape)

# Initialize the logistic regression model with L1 regularization
model = LogisticRegression(penalty='l1', tol = 0.01, solver = 'liblinear')

# Train the model
model.fit(X_train_reduced, y_train)

# Evaluate the model
train_accuracy = model.score(X_train_reduced, y_train)
test_accuracy = model.score(X_test_reduced, y_test)

print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

plt.figure(figsize=(8, 6))
plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=y_train.values, cmap='viridis', alpha=0.7, label='Training Data')
plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=y_test.values, cmap='viridis', alpha=0.7, edgecolors='k', label='Test Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization (Colored by Class Labels)')
plt.legend()
plt.grid(True)
plt.show()