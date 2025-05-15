{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d067280-1f96-4665-a25f-8035910693ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Project Research Question:\n",
    "#----------------------------------------------\n",
    "##**Can we build a machine learning model to predict whether a patient is experiencing an epileptic seizure based on EEG signals?**##"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "122c7adc-bb49-43b2-916c-8dfbb08c7375",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.feature_selection import SelectKBest, f_classif, RFE\n",
    "from sklearn.metrics import (accuracy_score, classification_report, \n",
    "                           confusion_matrix, roc_curve, auc, \n",
    "                           precision_recall_curve, average_precision_score)\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.pipeline import Pipeline\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from collections import Counter\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ee6b096-af91-4965-aa27-1afe4819db08",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "df = pd.read_csv('Epileptic Seizure Recognition.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9d32179-1d19-489c-98ae-a9d59d430fd9",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03657db2-4bd6-44b4-a660-cc567a1630e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check first few rows\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "598361bd-9e88-4a4a-b163-f6f689820f54",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop the ID column\n",
    "df = df.drop(['Unnamed'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93c6a0cd-a97e-4f9f-9235-a3a22e7e8d38",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check missing values\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc17f7e4-b741-4049-bb4c-64182ace97f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Describe numerical features\n",
    "print(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ac5db03d-0457-4503-8573-96565a5a7894",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Handle missing values (replace NaN with column mean)\n",
    "df.fillna(df.mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c26d33bf-05f3-444a-8cc5-8c5a872c4358",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verify missing values are handled\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d67a095-38d4-4644-9156-04b756651684",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assuming EEG data is in columns like 'X1', 'X2', ..., 'X178'\n",
    "signal_cols = [col for col in df.columns if col.startswith('X')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1debda01-6ff3-482b-844a-f57305088b8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create new feature: mean EEG signal\n",
    "df['mean_signal'] = df[signal_cols].mean(axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55809cc3-7be3-4a5d-9dac-4a7e382aafc2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# histogram\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.histplot(df[df.columns[30:50]], kde=True, bins=30)\n",
    "plt.title(f'Histogram of {df.columns[0]}')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5882bc2-9906-4c8e-abf4-8468a5495106",
   "metadata": {},
   "outputs": [],
   "source": [
    "# heatmap\n",
    "plt.figure(figsize=(100,50))\n",
    "corr = df.corr()\n",
    "sns.heatmap(corr, annot=True, fmt=\".2f\", cmap='coolwarm')\n",
    "plt.title('Correlation Heatmap')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42c77e4a-3b53-47f3-a39b-2fb3d171e11a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#box plot\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(data=df.iloc[:, 1:50])  \n",
    "plt.title(\"EEG Signal Distributions\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6253a3db-a085-490a-85be-88d602535596",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pca plot\n",
    "from sklearn.decomposition import PCA\n",
    "pca = PCA(n_components=2)\n",
    "X_pca = pca.fit_transform(df.drop('y', axis=1))\n",
    "sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['y'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "803d283e-7b1c-4842-87d4-65ba3371a575",
   "metadata": {},
   "outputs": [],
   "source": [
    "#signals plot\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(df.iloc[0, 1:50])  \n",
    "plt.title(\"Sample EEG Signal\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de62a47f-bc8a-4e98-9d01-f45229b1dd20",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target Variable Analysis (assuming 'y' is target)\n",
    "print(\"\\nTarget variable distribution:\")\n",
    "print(df['y'].value_counts(normalize=True))\n",
    "\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.countplot(x='y', data=df)\n",
    "plt.title(\"Distribution of Target Variable\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6da649b2-f921-44f4-a516-e9941a841f40",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Correlation Analysis\n",
    "plt.figure(figsize=(12,8))\n",
    "corr = df.corr()\n",
    "sns.heatmap(corr, cmap='coolwarm', center=0)\n",
    "plt.title(\"Feature Correlation Heatmap\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d04a1e6-ed23-4ecf-93fa-fae5c4031221",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Feature Distributions (univariate)\n",
    "num_cols = df.select_dtypes(include=['int64','float64']).columns\n",
    "for col in num_cols[:5]:  # First 5 numerical features\n",
    "    plt.figure(figsize=(8,4))\n",
    "    sns.histplot(df[col], kde=True)\n",
    "    plt.title(f\"Distribution of {col}\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "834636f1-a15e-494c-bc56-6329dbd7c812",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Feature vs Target Analysis (bivariate)\n",
    "for col in num_cols[:3]:  # First 3 features vs target\n",
    "    plt.figure(figsize=(8,4))\n",
    "    sns.boxplot(x='y', y=col, data=df)\n",
    "    plt.title(f\"{col} by Target Class\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f80c6619-691b-40c6-84ca-102605094972",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Outlier Detection\n",
    "for col in num_cols[:3]:\n",
    "    plt.figure(figsize=(8,4))\n",
    "    sns.boxplot(x=df[col])\n",
    "    plt.title(f\"Boxplot of {col}\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f25ae988-5a78-460c-a0b7-e2334da17b30",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Time Series Visualization (for EEG data)\n",
    "plt.figure(figsize=(12,6))\n",
    "sample_idx = 0  # First sample\n",
    "plt.plot(df.iloc[sample_idx, :-1])  # All features except target\n",
    "plt.title(f\"EEG Signal Sample (Class: {df.iloc[sample_idx, -1]})\")\n",
    "plt.xlabel(\"Time Points\")\n",
    "plt.ylabel(\"Signal Value\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26305a9a-1b2e-446f-9717-118d26d56a77",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assuming y=1 is seizure and others are non-seizure\n",
    "df['binary_label'] = df['y'].apply(lambda x: 1 if x == 1 else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e21524b9-6d6d-41b1-8b2b-64f4cca9ecee",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Separate features and target\n",
    "X = df.drop('y', axis=1)  \n",
    "y = df['y']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5fd86194-b4bf-4ad8-9003-c84407b47da1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove constant features\n",
    "X_non_constant = X.loc[:, (X != X.iloc[0]).any(axis=0)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b3a3224-45e3-4c60-89df-4ce1356e3962",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Scale the features\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d56ec87a-f2fe-4a21-9ba8-77e447f9962b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature selection: Remove features with very low variance\n",
    "from sklearn.feature_selection import VarianceThreshold\n",
    "variance_threshold = 0.01  \n",
    "selector_variance = VarianceThreshold(threshold=variance_threshold)\n",
    "X_non_constant = selector_variance.fit_transform(X_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2e3115b-da75-4747-ac5d-4b22443e8460",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check the feature variances after variance thresholding\n",
    "variances = np.var(X_non_constant, axis=0)\n",
    "print(f\"Min Variance: {np.min(variances)} | Max Variance: {np.max(variances)} | Mean Variance: {np.mean(variances)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd8aad65-6aac-42c8-b680-e074ffe89384",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scale the features after removing constant and low-variance features\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X_non_constant) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "386e29b3-512d-4246-afa7-daa722f721a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature selection: select top 20 features based on ANOVA F-test\n",
    "selector_kbest = SelectKBest(score_func=f_classif, k=20)  \n",
    "X_selected = selector_kbest.fit_transform(X_non_constant, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c52765d-4b57-4106-a3f0-6738409cf618",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1fb4bca3-36c9-440e-b329-0dbc5401806b",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"Shape of X_train: {X_train.shape} | Shape of X_test: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b4b13d3d-470d-4ba5-8c2d-19a219bdb2ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "# New feature: Mean EEG signal per sample\n",
    "df['EEG_mean'] = df.iloc[:, 1:179].mean(axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fbd0642-9051-4305-8a63-caa2e980076d",
   "metadata": {},
   "outputs": [],
   "source": [
    "correlations = df.corr()['y'].abs().sort_values(ascending=False)\n",
    "top_features = correlations[1:11].index  # Top 10 features\n",
    "X = df[top_features]\n",
    "y = df['y']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa2db06a-8d60-4d69-b632-9bebc4ce1949",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Initialize models\n",
    "models = {\n",
    "    'SVM': SVC(random_state=42),\n",
    "    'DecisionTree': DecisionTreeClassifier(random_state=42),\n",
    "    'KNN': KNeighborsClassifier(),\n",
    "    'RandomForest': RandomForestClassifier(random_state=42),\n",
    "    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "819dff59-e502-4c2e-8f13-8ede9add8f34",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score\n",
    "from sklearn.preprocessing import label_binarize\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import roc_curve, auc\n",
    "\n",
    "# Define models dictionary with models (ensure to set probability=True for SVC)\n",
    "models = {\n",
    "    'SVC': SVC(probability=True),  \n",
    "    \n",
    "}\n",
    "\n",
    "# Train + Evaluate\n",
    "for name, model in models.items():\n",
    "    # Fit the model on the training data\n",
    "    model.fit(X_train, y_train)\n",
    "    \n",
    "    # Make predictions\n",
    "    y_pred = model.predict(X_test)\n",
    "    \n",
    "    # Check if the model supports predict_proba\n",
    "    if hasattr(model, 'predict_proba'):\n",
    "        y_proba = model.predict_proba(X_test)  \n",
    "    else:\n",
    "        y_proba = None  \n",
    "\n",
    "    # Metrics\n",
    "    print(f\"\\n{name}\")\n",
    "    print(\"Confusion Matrix:\\n\", confusion_matrix(y_test, y_pred))\n",
    "    print(\"Classification Report:\\n\", classification_report(y_test, y_pred))\n",
    "    print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "\n",
    "    # Plot Confusion Matrix\n",
    "    plt.figure(figsize=(5,4))\n",
    "    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')\n",
    "    plt.title(f'Confusion Matrix - {name}')\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.ylabel('Actual')\n",
    "    plt.show()\n",
    "\n",
    "    # If the model supports `predict_proba`, plot ROC Curve\n",
    "    if y_proba is not None:\n",
    "        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))  # Binarize the output labels\n",
    "\n",
    "        # For each class, plot the ROC curve\n",
    "        plt.figure(figsize=(6,5))\n",
    "        for i in range(y_test_bin.shape[1]):\n",
    "            fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_proba[:, i])\n",
    "            roc_auc = auc(fpr, tpr)\n",
    "            plt.plot(fpr, tpr, label=f'Class {i} ROC Curve (AUC = {roc_auc:.2f})')\n",
    "\n",
    "        plt.plot([0,1], [0,1], linestyle='--', color='gray')\n",
    "        plt.xlabel('False Positive Rate')\n",
    "        plt.ylabel('True Positive Rate')\n",
    "        plt.title(f'ROC Curve - {name}')\n",
    "        plt.legend()\n",
    "        plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb65084e-d0fd-4d8f-af22-a0139857f90d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#GridSearchCV for SVM\n",
    "param_grid_svm = {\n",
    "    'C': [0.1, 1, 10],\n",
    "    'kernel': ['linear', 'rbf'],\n",
    "    'gamma': ['scale', 'auto']\n",
    "}\n",
    "\n",
    "grid_search_svm = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=5, scoring='accuracy')\n",
    "grid_search_svm.fit(X_train, y_train)\n",
    "\n",
    "print(\"Best SVM Parameters\")\n",
    "print(grid_search_svm.best_params_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ceafd1f-c4b4-434f-a394-e47b8d19d3a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Evaluate tuned SVM\n",
    "best_svm = grid_search_svm.best_estimator_\n",
    "best_pred = best_svm.predict(X_test)\n",
    "best_proba = best_svm.predict_proba(X_test)[:,1]\n",
    "\n",
    "print(\"\\n=== Tuned SVM Results ===\")\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_test, best_pred))\n",
    "print(\"Classification Report:\\n\", classification_report(y_test, best_pred))\n",
    "print(\"Accuracy:\", accuracy_score(y_test, best_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41bb924c-5398-4176-ad0f-4cc3f7af9d5a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Plot Confusion Matrix for tuned SVM\n",
    "plt.figure(figsize=(5,4))\n",
    "sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt='d', cmap='Greens')\n",
    "plt.title('Confusion Matrix - Tuned SVM')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4325703b-cb5f-48b8-a099-b209376919b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#SVM model after hyperparameter tuning\n",
    "svm_model = SVC(probability=True)\n",
    "\n",
    "# Train your model\n",
    "svm_model.fit(X_train, y_train)\n",
    "\n",
    "# Get probabilities for each class\n",
    "best_proba = svm_model.predict_proba(X_test)  \n",
    "\n",
    "# Check the shape of best_proba\n",
    "print(best_proba.shape)  \n",
    "\n",
    "#proceed to plot the ROC curve\n",
    "from sklearn.metrics import roc_curve, auc\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Binarize the output labels (for multiclass ROC)\n",
    "y_test_bin = label_binarize(y_test, classes=np.unique(y_test))\n",
    "\n",
    "# Plot ROC curve for each class\n",
    "plt.figure(figsize=(6, 5))\n",
    "for i in range(y_test_bin.shape[1]):  \n",
    "    fpr_best, tpr_best, _ = roc_curve(y_test_bin[:, i], best_proba[:, i])  \n",
    "    roc_auc_best = auc(fpr_best, tpr_best)\n",
    "    plt.plot(fpr_best, tpr_best, label=f'Class {i} ROC Curve (AUC = {roc_auc_best:.2f})')\n",
    "\n",
    "# Plot the diagonal line for random chance\n",
    "plt.plot([0, 1], [0, 1], linestyle='--', color='gray')\n",
    "\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC Curve - Tuned SVM')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8faf6f51-69cc-4a7d-b92e-022a32937f21",
   "metadata": {},
   "outputs": [],
   "source": [
    "#logestic regression\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "# Initialize\n",
    "log_reg = LogisticRegression(max_iter=1000, random_state=42)\n",
    "\n",
    "# Hyperparameter tuning\n",
    "param_grid = {\n",
    "    'C': [0.001, 0.01, 0.1, 1, 10],\n",
    "    'penalty': ['l1', 'l2'],\n",
    "    'solver': ['liblinear']\n",
    "}\n",
    "grid_log = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1')\n",
    "grid_log.fit(X_train, y_train)\n",
    "\n",
    "# Best model\n",
    "best_log = grid_log.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "04ecf107-df12-4d01-9ecc-1590a4d8a2fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "#random forest\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Initialize\n",
    "rf = RandomForestClassifier(random_state=42)\n",
    "\n",
    "# Hyperparameter tuning\n",
    "param_grid = {\n",
    "    'n_estimators': [50, 100, 200],\n",
    "    'max_depth': [None, 10, 20],\n",
    "    'min_samples_split': [2, 5]\n",
    "}\n",
    "grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='f1')\n",
    "grid_rf.fit(X_train, y_train)\n",
    "\n",
    "# Best model\n",
    "best_rf = grid_rf.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cb2dd35-4429-474f-9f0e-f0277839304e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#SVC\n",
    "from sklearn.svm import SVC\n",
    "\n",
    "# Initialize\n",
    "svm = SVC(probability=True, random_state=42)\n",
    "\n",
    "# Hyperparameter tuning\n",
    "param_grid = {\n",
    "    'C': [0.1, 1, 10],\n",
    "    'kernel': ['linear', 'rbf'],\n",
    "    'gamma': ['scale', 'auto']\n",
    "}\n",
    "grid_svm = GridSearchCV(svm, param_grid, cv=3, scoring='f1')  # Smaller cv due to SVM's slowness\n",
    "grid_svm.fit(X_train, y_train)\n",
    "\n",
    "# Best model\n",
    "best_svm = grid_svm.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fe292f2c-13f9-4c30-a6d9-526ea7eec256",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def evaluate_model(model, X_test, y_test, name):\n",
    "    y_pred = model.predict(X_test)\n",
    "    \n",
    "    print(f\"\\n=== {name} ===\")\n",
    "    print(\"Best Params:\", grid_search.best_params_ if 'grid_' in name.lower() else \"N/A\")\n",
    "    print(classification_report(y_test, y_pred))\n",
    "    \n",
    "    # ROC-AUC calculation for multi-class\n",
    "    if hasattr(model, \"predict_proba\"):\n",
    "        y_proba = model.predict_proba(X_test)\n",
    "        # For multi-class, we need to specify the multi_class parameter\n",
    "        try:\n",
    "            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')\n",
    "            print(f\"ROC-AUC (OVR): {roc_auc:.3f}\")\n",
    "        except ValueError:\n",
    "            # If binary classification or other error\n",
    "            print(\"Could not calculate ROC-AUC with OVR method\")\n",
    "    else:\n",
    "        print(\"Model doesn't support predict_proba, skipping ROC-AUC\")\n",
    "    \n",
    "    # Confusion Matrix\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    plt.figure(figsize=(8, 6))\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
    "    plt.title(f\"{name} Confusion Matrix\")\n",
    "    plt.xlabel('Predicted labels')\n",
    "    plt.ylabel('True labels')\n",
    "    plt.show()\n",
    "\n",
    "# Evaluate all models\n",
    "evaluate_model(best_log, X_test, y_test, \"Logistic Regression (Tuned)\")\n",
    "evaluate_model(best_rf, X_test, y_test, \"Random Forest (Tuned)\")\n",
    "evaluate_model(best_svm, X_test, y_test, \"SVM (Tuned)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "346fd49e-0e99-4cb9-84f6-03f9614aa93a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import (\n",
    "    RandomForestClassifier, AdaBoostClassifier, VotingClassifier\n",
    ")\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.neighbors import KNeighborsClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62af013a-8c87-4782-b8fa-4b13cb256803",
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(max_iter=1000),\n",
    "    'K-Nearest Neighbors': KNeighborsClassifier(),\n",
    "    'Naive Bayes': GaussianNB()\n",
    "}\n",
    "\n",
    "results = {}\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    results[name] = accuracy_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "163c75a2-f79d-4928-a505-b68af3f4dc66",
   "metadata": {},
   "outputs": [],
   "source": [
    "rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf.fit(X_train, y_train)\n",
    "results['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ae73ca4-22a1-425d-a54a-0270b336aad9",
   "metadata": {},
   "outputs": [],
   "source": [
    "ab = AdaBoostClassifier(n_estimators=50, random_state=42)\n",
    "ab.fit(X_train, y_train)\n",
    "results['AdaBoost'] = accuracy_score(y_test, ab.predict(X_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5da7d2ee-ca17-4fbe-b1b4-3bf0927805c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "voting = VotingClassifier(estimators=[\n",
    "    ('lr', LogisticRegression(max_iter=1000)),\n",
    "    ('knn', KNeighborsClassifier()),\n",
    "    ('nb', GaussianNB())\n",
    "], voting='hard')\n",
    "\n",
    "voting.fit(X_train, y_train)\n",
    "results['Voting Ensemble'] = accuracy_score(y_test, voting.predict(X_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6197368c-2110-490d-b12b-bb4826da96bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "results = {}\n",
    "reports = {}\n",
    "conf_matrices = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "\n",
    "    # Accuracy\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    results[name] = acc\n",
    "\n",
    "    # Classification report (dictionary format for later use)\n",
    "    reports[name] = classification_report(y_test, y_pred, output_dict=True)\n",
    "\n",
    "    # Confusion matrix\n",
    "    conf_matrices[name] = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "# Summary table with Accuracy\n",
    "results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])\n",
    "print(\"\\n Accuracy Summary:\")\n",
    "print(results_df)\n",
    "\n",
    "# Detailed classification reports\n",
    "for name, report in reports.items():\n",
    "    print(f\"\\nClassification Report for {name}:\\n\")\n",
    "    print(classification_report(y_test, models[name].predict(X_test)))\n",
    "\n",
    "# Confusion matrices\n",
    "for name, matrix in conf_matrices.items():\n",
    "    plt.figure(figsize=(4, 3))\n",
    "    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')\n",
    "    plt.title(f'Confusion Matrix: {name}')\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.ylabel('True')\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0564c5af-101f-440a-be30-813fc31c8987",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import VotingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Define models\n",
    "lr = LogisticRegression(max_iter=1000)\n",
    "nb = GaussianNB()\n",
    "knn = KNeighborsClassifier()\n",
    "\n",
    "# Hard Voting\n",
    "voting_hard = VotingClassifier(\n",
    "    estimators=[('lr', lr), ('nb', nb), ('knn', knn)],\n",
    "    voting='hard'\n",
    ")\n",
    "\n",
    "# Soft Voting\n",
    "voting_soft = VotingClassifier(\n",
    "    estimators=[('lr', lr), ('nb', nb), ('knn', knn)],\n",
    "    voting='soft'\n",
    ")\n",
    "\n",
    "#Train & Evaluate Hard Voting \n",
    "voting_hard.fit(X_train, y_train)\n",
    "y_pred_hard = voting_hard.predict(X_test)\n",
    "print(\" Hard Voting Accuracy:\", accuracy_score(y_test, y_pred_hard))\n",
    "print(\" Classification Report (Hard Voting):\")\n",
    "print(classification_report(y_test, y_pred_hard))\n",
    "\n",
    "plt.figure(figsize=(4, 3))\n",
    "sns.heatmap(confusion_matrix(y_test, y_pred_hard), annot=True, fmt='d', cmap='Blues')\n",
    "plt.title(' Confusion Matrix - Hard Voting')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('True')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "#Train & Evaluate Soft Voting\n",
    "voting_soft.fit(X_train, y_train)\n",
    "y_pred_soft = voting_soft.predict(X_test)\n",
    "print(\"\\n Soft Voting Accuracy:\", accuracy_score(y_test, y_pred_soft))\n",
    "print(\" Classification Report (Soft Voting):\")\n",
    "print(classification_report(y_test, y_pred_soft))\n",
    "\n",
    "plt.figure(figsize=(4, 3))\n",
    "sns.heatmap(confusion_matrix(y_test, y_pred_soft), annot=True, fmt='d', cmap='Oranges')\n",
    "plt.title(' Confusion Matrix - Soft Voting')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('True')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99e30fa7-35ff-4bc8-ad95-4d73196debce",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import BaggingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "bagging_model = BaggingClassifier(\n",
    "    estimator=LogisticRegression(max_iter=1000),\n",
    "    n_estimators=10,\n",
    "    random_state=42\n",
    ")\n",
    "bagging_model.fit(X_train, y_train)\n",
    "print(\"Bagging Accuracy:\", bagging_model.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86747942-e3be-4479-be34-f6519f8da828",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "boosting_model = AdaBoostClassifier(\n",
    "    estimator=DecisionTreeClassifier(max_depth=2),\n",
    "    n_estimators=50,\n",
    "    learning_rate=1.0,\n",
    "    random_state=42\n",
    ")\n",
    "boosting_model.fit(X_train, y_train)\n",
    "print(\"Boosting Accuracy:\", boosting_model.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9c34c4c-0ad7-4744-8dc3-b92fda17c7b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import StackingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Define base learners\n",
    "base_models = [\n",
    "    ('lr', LogisticRegression(max_iter=1000)),\n",
    "    ('knn', KNeighborsClassifier()),\n",
    "    ('nb', GaussianNB())\n",
    "]\n",
    "\n",
    "# Meta-learner (final decision)\n",
    "stacking_model = StackingClassifier(\n",
    "    estimators=base_models,\n",
    "    final_estimator=LogisticRegression(),\n",
    "    cv=5\n",
    ")\n",
    "\n",
    "stacking_model.fit(X_train, y_train)\n",
    "print(\"Stacking Accuracy:\", stacking_model.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "446b8784-012e-47f7-844f-a27feb415f56",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define models\n",
    "models = {\n",
    "    \"Bagging (LogReg)\": BaggingClassifier(estimator=LogisticRegression(max_iter=1000), n_estimators=10, random_state=42),\n",
    "    \"Boosting (Tree)\": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50, random_state=42),\n",
    "    \"Stacking (LR+KNN+NB)\": StackingClassifier(\n",
    "        estimators=[\n",
    "            ('lr', LogisticRegression(max_iter=1000)),\n",
    "            ('knn', KNeighborsClassifier()),\n",
    "            ('nb', GaussianNB())\n",
    "        ],\n",
    "        final_estimator=LogisticRegression(),\n",
    "        cv=5\n",
    "    ),\n",
    "    \"Voting (LR+KNN+NB)\": VotingClassifier(\n",
    "        estimators=[\n",
    "            ('lr', LogisticRegression(max_iter=1000)),\n",
    "            ('knn', KNeighborsClassifier()),\n",
    "            ('nb', GaussianNB())\n",
    "        ],\n",
    "        voting='hard'\n",
    "    )\n",
    "}\n",
    "\n",
    "# Collect metrics\n",
    "summary = []\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    report = classification_report(y_test, y_pred, output_dict=True)\n",
    "    summary.append({\n",
    "        \"Model\": name,\n",
    "        \"Accuracy\": accuracy_score(y_test, y_pred),\n",
    "        \"Precision\": report['weighted avg']['precision'],\n",
    "        \"Recall\": report['weighted avg']['recall'],\n",
    "        \"F1-score\": report['weighted avg']['f1-score']\n",
    "    })\n",
    "\n",
    "df_summary = pd.DataFrame(summary)\n",
    "print(df_summary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "38dea997-392c-4c03-a4b5-de09ae591fd7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "\n",
    "# Your full model comparison data\n",
    "df = pd.DataFrame({\n",
    "    \"Model\": [\"Bagging (LogReg)\", \"Boosting (Tree)\", \"Stacking (LR+KNN+NB)\", \"Voting (LR+KNN+NB)\"],\n",
    "    \"Accuracy\": [0.408696, 0.490870, 0.530870, 0.463913],\n",
    "    \"Precision\": [0.405603, 0.495849, 0.530129, 0.475394],\n",
    "    \"Recall\": [0.408696, 0.490870, 0.530870, 0.463913],\n",
    "    \"F1-score\": [0.405395, 0.476045, 0.529224, 0.455507],\n",
    "})\n",
    "\n",
    "# Reshape the DataFrame for seaborn\n",
    "df_melted = df.melt(id_vars=\"Model\", var_name=\"Metric\", value_name=\"Score\")\n",
    "\n",
    "# Plot everything\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(data=df_melted, x=\"Model\", y=\"Score\", hue=\"Metric\", palette=\"Set2\")\n",
    "plt.title(\"Performance Comparison of Ensemble Models\")\n",
    "plt.xticks(rotation=15)\n",
    "plt.ylim(0, 1)\n",
    "plt.legend(title=\"Metric\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1f1f347-5184-4a6f-be12-9537a70bac9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# If X_train is a numpy array, convert it to a pandas DataFrame\n",
    "X_train_df = pd.DataFrame(X_train)\n",
    "X_train_df.columns = [f'Feature_{i}' for i in range(X_train_df.shape[1])]\n",
    "\n",
    "# Feature Importance - Random Forest\n",
    "rf_importances = pd.DataFrame({\n",
    "    'Feature': X_train_df.columns,\n",
    "    'Importance': best_rf.feature_importances_\n",
    "}).sort_values(by='Importance', ascending=False)\n",
    "\n",
    "# Display top 20 features if there are many\n",
    "top_n = min(20, len(rf_importances))\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.barplot(data=rf_importances.head(top_n), x='Importance', y='Feature', palette='Blues_r')\n",
    "plt.title('Top Feature Importance - Random Forest')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
