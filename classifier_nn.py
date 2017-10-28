
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.callbacks import History
#from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import sys
from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold

def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def get_traintest_validation_split(df):
    """ Split the data into a train_test dataset and a validation dataset
        train_test will later be split into the training and testing
        dataset
        Fitting using gradient descent etc is only performed on the
        training set, fitting via the act of hyperparameter optimisation
        uses the training and test dataset as tuners (this inherently fits
        the data to the training and test sets).
        This is the reason that the test dataset is not sufficient as a
        validation set
        (it can result in overestimated performance as the current optimised
         model has been fine-tuned to performed well on the test set)
         This is why the final model is tested on a hold-out validation set
    """
    train_test, validation = np.split(df.sample(frac=1),
                                      [int(.9 * len(df))])
    y = np.ravel(train_test['Class'])
    train_test = train_test.drop('Class', 1)
    x = train_test.as_matrix()
    return x, y, validation


def set_plots():
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    plt.title('Learning Rate Profile')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.title('Accuracy Evolution')
    ax2 = fig2.add_subplot(111)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Loss")
    plt.title('Learning Rate Profile - valid')
    ax3 = fig3.add_subplot(111)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Accuracy")
    plt.title('Accuracy Evolution - valid')
    ax4 = fig4.add_subplot(111)
    return ax1, ax2, ax3, ax4


def manual_ann(X, y, validation):
    """ Keras Sequential model with Manual validation
    """
    # fix random seed for reproducibility
    seed = 13
    np.random.seed(seed)
    cvscores = []

    # NN architecture
    input_nodes = 75
    hidden1_nodes = 50
    hidden2_nodes = 50
    hidden3_nodes = 50
    output_nodes = 1

    ax1, ax2, ax3, ax4 = set_plots()
    # Split with ratio 70:30 for train/test
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=seed)
    # create model
    dropout_rate1 = 0.1
    dropout_rate2 = 0.1
    learning_rate = 0.2
    sgd = SGD(lr=learning_rate, momentum=0.05, nesterov=False)
    # create Sequential object with spec
    model = Sequential()
    model.add(Dense(output_dim=hidden1_nodes,
                    input_dim=input_nodes,
                    activation='relu'))
    model.add(Dropout(dropout_rate1))
    model.add(Dense(output_dim=hidden2_nodes,
                    input_dim=hidden1_nodes,
                    activation='relu'))
    model.add(Dropout(dropout_rate2))
    model.add(Dense(output_dim=hidden3_nodes,
                    input_dim=hidden2_nodes,
                    activation='relu'))
    model.add(Dropout(dropout_rate2))
    model.add(Dense(output_dim=output_nodes,
                    input_dim=hidden3_nodes,
                    activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    history = History()

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # Fit the model
    history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=200,
                        verbose=2)
    # evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
    set_trace()

    # Fit the model
    grid_result = model.fit(X_train,
                            y_train,
                            validation_data=[X_test, y_test],
                            epochs=100,
                            batch_size=200,
                            verbose=2,
                            callbacks=[history])
    ax1.plot(history.history["loss"], label='Train-Loss')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.plot(history.history["acc"], label='Train-Acc')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.plot(history.history["val_loss"], label='Val-Loss')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax4.plot(history.history["val_acc"], label='Val-Acc')
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot the validation accuracy on the same plot as training accuracy
    ax2.plot(history.history["val_acc"], label='Val-Acc')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    set_trace()
    # Convert validity set to arrays, centre and normalise and fit/evaluate accuracy
    cvscores = []
    set_trace()
    '''
    y_val = np.ravel(validation['Churn Flag'])
    X_val = validation[x_cols]
    X_val = X_val.as_matrix()
    scale = StandardScaler()
    X_val = scale.fit_transform(X_val)
    scores = model.evaluate(X_val, y_val, verbose=2)
    print"Results of manual validation %s: %.2f%%" % (model.metrics_names[1],
                                                      scores[1] * 100)
    cvscores.append(scores[1] * 100)
    print"%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))
    probabilities = model.predict_proba(X_val, verbose=2)
    class_prediction = model.predict_classes(X_val, verbose=2)
    '''
    return probabilities, class_prediction, X_val, y_val, history, cvscores


def ann_with_dropout(X, y):
    """ Keras Sequential model with Manual validation -
        Optimise Dropout (GridSearch)
    """
    # fix random seed for reproducibility
    seed = 14
    np.random.seed(seed)

    # NN architecture
    input_nodes = 50
    hidden1_nodes = 50
    hidden2_nodes = 50
    hidden3_nodes = 50
    output_nodes = 1

    # Split with ratio 70:30 for train/test
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=seed)

    # create model
    dropout_rate = [0.05, 0.1, 0.2, 0.4, 0.5, 0.8]
    # dropout_rate = 0.1
    learning_rate = 0.01
    sgd = SGD(lr=learning_rate, nesterov=False)

    def create_model(dropout_rate=0):
        model = Sequential()
        model.add(Dense(output_dim=hidden1_nodes,
                        input_dim=input_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=hidden2_nodes,
                        input_dim=hidden1_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=hidden3_nodes,
                        input_dim=hidden2_nodes,
                        activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim=output_nodes,
                        input_dim=hidden3_nodes,
                        activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model

    # create model
    model = KerasClassifier(build_fn=create_model,
                            epochs=100,
                            batch_size=200,
                            verbose=2)
    model.fit(X_train,
              y_train,
              validation_data=[X_test, y_test],
              epochs=100,
              batch_size=200,
              verbose=2)
    param_grid = dict(dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    # Fit the model
    grid_result = grid.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_,
                                 grid_result.best_params_))


def ann_k_fold_validation(X, y):
    """ K-fold Cross-validation
        K-fold cross-validation in general gives the most pessimistic accuracy.
        10-fold on 100,000 rows uses 9 sets of 10,000 to predict the hold-out set and gives each of the
        ten subsets a chance to be the hold-out set. Training and test sets are smaller (less predictive power) than
        a 70:30 split on the wholw set(70,000, 30,000)
    """
    # fix random seed for reproducibility
    seed = 13
    np.random.seed(seed)
    # NN architecture
    input_nodes = 75
    hidden1_nodes = 50
    hidden2_nodes = 50
    hidden3_nodes = 50
    output_nodes = 1

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    cvscores = []
    ax1, ax2, ax3, ax4 = set_plots()
    fold_count = 0
    for train, test in kfold.split(X, y):
        fold_count += 1
        # create model
        # model = Sequential()
        # model.add(Dense(12, input_dim=14, activation='relu'))
        # model.add(Dense(8, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))

        model = Sequential()
        model.add(Dense(output_dim=hidden1_nodes,
                        input_dim=input_nodes,
                        activation='relu'))
        model.add(Dense(output_dim=hidden2_nodes,
                        input_dim=hidden1_nodes,
                        activation='relu'))
        model.add(Dense(output_dim=hidden3_nodes,
                        input_dim=hidden2_nodes,
                        activation='relu'))
        model.add(Dense(output_dim=output_nodes,
                        input_dim=hidden3_nodes,
                        activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # Fit the model
        history = model.fit(X[train],
                            y[train],
                            epochs=100,
                            batch_size=200,
                            verbose=2)
        # evaluate the model
        scores = model.evaluate(X[test], y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

        ax1.plot(history.history["loss"], label='Loss %d' % fold_count)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.plot(history.history["acc"], label='Acc %d' % fold_count)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # ax3.plot(history.history["val_loss"], label='Loss %d' % fold_count)
        # ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax4.plot(history.history["val_acc"], label='Acc %d' % fold_count)
        # ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Plot the validation accuracy on the same plot as training accuracy
        # ax2.plot(history.history["val_acc"], label='Val-Acc %d' % fold_count)
        # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



def perform_logistic_regression(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    from sklearn.cross_validation import cross_val_score

    print ('The class balance is Criminal: {}%, Non Criminal: {}%'.
           format(round(y.mean() * 100), round((1 - y.mean())) * 100))

    # Training and testing on the same set
    model = LogisticRegression()
    model.fit(X, y)
    score1 = model.score(X, y)
    print 'Training and testing on the same set gives an accuracy of {}%'.format(score1 * 100)

    # Model evaluation using a validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)
    # Predict class labels for the test set
    predicted = model2.predict(X_test)
    # Generate class probabilities
    probs = model2.predict_proba(X_test)
    # generate evaluation metrics
    print ('Testing on a seperate validation set to training set gave an accuracy of: {}%'.
           format(round(metrics.accuracy_score(y_test, predicted)) * 100))
    print 'ROC_AUC score: {}'.format(metrics.roc_auc_score(y_test, probs[:, 1]))
    print 'Confusion matrix:\n {}'.format(metrics.confusion_matrix(y_test, predicted))
    print 'Classification report:\n {}'.format(metrics.classification_report(y_test, predicted))

    # Evaluate the model using 10-fold cross-validation
    scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print 'Performing 10-fold cross-validation gave an accuracy of:\n {}'.format(scores)
    print 'Mean 10-fold cross-validation socre: {}'.format(scores.mean())