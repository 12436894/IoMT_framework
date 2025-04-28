<!-- 🌟 Hybrid Machine Learning Model: Random Forest + Gradient Boosting -->
This project builds a hybrid machine learning model by combining Random Forest and Gradient Boosting classifiers into a Voting Classifier. The model predicts labels from a given dataset and provides detailed graphical visualizations of the results.
All outputs (evaluation reports, plots, datasets) are saved into an output/ directory.

<!-- 📚 Libraries Used -->

pandas
scikit-learn
matplotlib
seaborn

<!-- 📦 Install Required Libraries -->
bashpip install pandas scikit-learn matplotlib seaborn
<!-- How It Works -->

Load Dataset: Reads a CSV file into a pandas DataFrame.
<!-- Preprocessing -->

Separates features (X) and labels (y).
Applies Principal Component Analysis (PCA) for feature selection (optional but recommended).
One-hot encodes any categorical features if necessary.


Train-Test Split: Divides the data into training and testing sets.
<!-- Model Building: Combines: -->

Random Forest Classifier
Gradient Boosting Classifier into a Voting Classifier.


Training: Trains the hybrid model on the training dataset.
Prediction: Predicts labels for the test dataset.
<!-- Evaluation -->

Accuracy score
Classification report
Confusion matrix
ROC Curve
Precision-Recall Curve
Feature Importances (if enabled)


<!-- Saving Outputs -->

Saves train and test datasets as CSV files.
Saves all plots as PNG images in the output/ folder.
Saves evaluation report in a text file.



<!-- 📊 Visualizations After running the script, the following visualizations are generated: -->

Confusion Matrix (output/confusion_matrix.png)
ROC Curve (output/roc_curve.png)
Precision-Recall Curve (output/precision_recall_curve.png)
(Optional) Feature Importance Chart

<!-- 🚀 Usage Clone this repository: -->
bashgit clone https://github.com/your-username/your-repository.git
cd your-repository
Add your dataset:

Place your CSV file inside a data/ folder (e.g., data/your_file.csv).

Run the Python script:
bashpython main.py
<!-- Check results: -->

Outputs are stored inside the output/ directory.

          
<!-- 📄 License -->
This project is licensed under the MIT License.