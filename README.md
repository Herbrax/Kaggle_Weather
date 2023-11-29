# Kaggle Competition 1 - Simo Hakim - 20096040


Avant d'exécuter le projet, assurez-vous que les paquets suivants sont installés :

- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- threading

Nous avons 3 modèles dans le code :

LogisticalRegression: A custom implementation of Logistic Regression.
LinearSVMClassifier: An import of scikit-learn LinearSVC implementation.
XGBoostClassifier: An import of XGBoost.

Pour entraîner et évaluer les modèles, exécutez les fonctions respectives qui effectueront l'optimisation des hyperparamètres via la recherche en grid search:

startLR(): Trains and evaluates the Logistic Regression model.
startSVM_Linear(): Trains and evaluates the Linear SVM model.
startXGBOOST(): Trains and evaluates the XGBoost model.

Vous pouvez exécuter vos propres hyperparamètres en les définissant comme ceci :

lr_model = LogisticalRegression(learning_rate=0.5, epochs=200)
lr_model.train(training_data, training_labels_one_hot)
predictions = lr_model.compute_predictions(validation_data)

svm_model = LinearSVMClassifier(C=1)
svm_model.train(split_train_normalized, split_train_label)
val_preds_svm = svm_model.compute_predictions(split_validation_normalized)

xgb_model = XGBoostClassifier(max_depth=0.3 eta=4, num_class=3)
xgb_model.train(split_train_normalized, split_train_label, split_validation_normalized, split_validation_label)
val_preds_xgb = xgb_model.compute_predictions(split_validation_normalized)
