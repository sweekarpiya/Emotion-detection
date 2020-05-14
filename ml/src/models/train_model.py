import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from configs import config
from sklearn.naive_bayes import MultinomialNB # Baseline model for 1st Iteration
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from features.build_features import feature_engineer

model_dict={
    'naive_bayes': MultinomialNB(),
    'svc': SVC()
}

def ml_model_training(model='naive_bayes', save_report=False):
    """Conducts training of the Machine learning model

    Conducts the training of machine learning model using either
    Naive_bayes or Support_vector_machine.

    Parameters
    ----------
    model: 'String'
            either "naive_bayes" or "svc"
    
    save_report: bool
            if true saves the classification report and heatmap in CSV file 
            if false just prints the report
    returns
    -------
    trained_model: Object
                Trained Sklearn's model 
    """
    features, enc_labels, labels = feature_engineer()
    trained_model = None
    print()
    try:
        trained_model = model_dict[model]
        X_train, X_test, y_train, y_test = train_test_split(features,
                                        enc_labels.ravel(),
                                        test_size=0.2)

        print('------------Starting Model Training------------')

        trained_model.fit(X_train, y_train)

        print('------------Training Completed---------------')
        predicted_label_train = trained_model.predict(X_train)
        predicted_label_test = trained_model.predict(X_test)

        if save_report==True:
            print('Saving the classification report')
            
            train_report = classification_report(y_train, predicted_label_train, digits=3, output_dict=True)
            test_report = classification_report(y_test, predicted_label_test, digits=3, output_dict=True)
            train_report = pd.DataFrame(train_report).transpose()

            test_report = pd.DataFrame(test_report).transpose()

            train_report.to_csv(os.path.join(config.OUTPUT_PATH,'train_report.csv'))
            test_report.to_csv(os.path.join(config.OUTPUT_PATH,'test_report.csv'))

            confusion_mat = confusion_matrix(y_test, predicted_label_test)
            ax = sns.heatmap(confusion_mat.T, square=True, annot=True, fmt='d', cbar=False)
            ax.set_xticklabels(list(labels[0]))
            ax.set_yticklabels(list(labels[0]))
            plt.savefig(os.path.join(config.OUTPUT_PATH,'heatmap.png'))
            print('Completed')

        else:
            print('***********For Training************')
            print(classification_report(y_train, predicted_label_train, digits=3))
            print('***********For Testing************')
            print(classification_report(y_test, predicted_label_test, digits=3))
        
    except KeyError:
        print('Please enter a valid model, either \'naive_bayes\' or \'svc\'')

    return trained_model

