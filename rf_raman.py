'''
Jade's 
Perform Undersampling and Smote
print the number of samples in each class before and after 
'''


import pandas as pd
import numpy as np
import os
from scipy.signal import savgol_filter
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

def read_file(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    toxicity = parts[1]
    patient_id = parts[2]
    sample_number = parts[3].split('.')[0]
    
    data = pd.read_csv(filepath, header=None, names=['Wavenumber', 'Intensity'])
    data['Toxicity'] = toxicity
    data['Patient_ID'] = patient_id
    data['Sample_Number'] = sample_number
    return data

def preprocess_data(data):
    data['Intensity'].fillna(method='ffill', inplace=True)
    data['Intensity'].fillna(method='bfill', inplace=True)
    data['Intensity'] = (data['Intensity'] - data['Intensity'].min()) / (data['Intensity'].max() - data['Intensity'].min())
    data['Intensity'] = savgol_filter(data['Intensity'], window_length=11, polyorder=2)
    return data

def combine_files(directory):
    all_data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                file_data = read_file(filepath)
                file_data = preprocess_data(file_data)
                all_data.append(file_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    final_data = combined_data.pivot_table(index=['Toxicity', 'Patient_ID', 'Sample_Number'],
                                           columns='Wavenumber', values='Intensity', aggfunc='first').reset_index()

    if 'index' in final_data.columns:
        final_data.drop('index', axis=1, inplace=True)
    return final_data

path_to_data = 'F:\\TUD\\Jade\\Sample_data\\'
final_data = combine_files(path_to_data)
final_tsv_filename = 'combined_data.tsv'
final_data.to_csv(final_tsv_filename, sep='\t', index=False)


'''
Random Forest
'''

data_path = 'combined_data.tsv'
data = pd.read_csv(data_path, sep='\t')
X = data.drop(['Toxicity', 'Patient_ID', 'Sample_Number'], axis=1)
y = data['Toxicity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train_smote, y_train_smote)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


'''
PCA-RF pipeline
'''

# Set the Seaborn style for all plots
sns.set(style="ticks", rc={
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.labelweight': 'bold',
    'font.size': 18,
    'font.weight': 'bold'
})

data_path = 'combined_data.tsv'
data = pd.read_csv(data_path, sep='\t')
X = data.drop(['Toxicity', 'Patient_ID', 'Sample_Number'], axis=1)
y = data['Toxicity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

pipeline = Pipeline([
    ('pca', PCA(n_components=20)),  # play with components
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4]  # i think 2 is having better result
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_smote, y_train_smote)
print("Best parameters found:", grid_search.best_params_)
best_model = grid_search.best_estimator_
cv = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
print("Cross-validation scores:", cross_val_scores)
print("Average CV score:", np.mean(cross_val_scores))
best_model.fit(X_train_smote, y_train_smote)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='g2')
roc_auc = auc(fpr, tpr)
plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Random Forest').plot()
plt.title('ROC Curve')
plt.show()

y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
