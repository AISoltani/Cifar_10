import numpy as np
import os
import matplotlib.pyplot as plt
from utility import plot_data
from preprocess import *
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import shuffle


def main():
    path = 'E:/Master/Semester 2/ML/Homeworks/Pure Code/ML/FinalProject/cifar-10-batches-py/'
    files_path = os.listdir(path)
    train_patches = files_path[1:6]
    test_patch = files_path[7]
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    num_train, num_test = 50000, 10000
    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for data_batch in train_patches:
        ds_batch = unpickle(path+data_batch)
        train_data += ds_batch[b'data'].tolist()
        train_labels += ds_batch[b'labels']

    ds_batch = unpickle(path+test_patch)
    test_data += ds_batch[b'data'].tolist()
    test_labels += ds_batch[b'labels']

    train_data = train_data[:num_train]
    train_labels = train_labels[:num_train]
    test_data = test_data[:num_test]
    test_labels = test_labels[:num_test]

    ds_title_preproc = 'Applied PLS-PCA on dataset'
    samples, labels = train_data, train_labels
    samples, labels = train_data+test_data, train_labels+test_labels
    # orginal_ds = samples

    # PCA Whitening
    # samples_pca_whitening = pca_whitening(train_data, test_data).tolist()
    # samples = samples_pca_whitening

    # PLS
    from sklearn.cross_decomposition import PLSRegression
    pls = PLSRegression(n_components=100)
    samples, _ = pls.fit_transform(samples, labels)
    # train_data = samples[:num_train]
    # test_data = samples[num_train:num_train+num_test]

    # PCA
    # samples_pca, _, _ = pca(train_data, test_data)
    samples_pca, _, _ = pca(samples)
    samples = samples_pca
    train_data = samples[:num_train]
    test_data = samples[num_train:num_train+num_test]





    # LDA
    # clf = LinearDiscriminantAnalysis()
    # samples = clf.fit_transform(samples, labels)
    # train_data = samples[:num_train]
    # test_data = samples[num_train:num_train+num_test]

    # HOG Features
    # samples_hog = simple_hog_features(train_data+test_data)
    # train_data = samples_hog[:num_train]
    # test_data = samples_hog[num_train:num_train+num_test]

    # SIFT Features
    # samples_sift = SIFT(train_data+test_data)
    # train_data = samples_sift[:num_train]
    # test_data = samples_sift[num_train:num_train+num_test]

    # Filter
    # samples_filter = filter_feature_selection(train_data+test_data, 100)
    # train_data = samples_filter[:num_train]
    # test_data = samples_filter[num_train:num_train+num_test]

    classifiers = []
    # KNN Classifier
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh_5 = KNeighborsClassifier(n_neighbors=5)
    neigh_9 = KNeighborsClassifier(n_neighbors=9)

    from sklearn import linear_model
    clf_sgd = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()

    from sklearn import tree
    DT_clf = tree.DecisionTreeClassifier()

    # Logistic regression
    from sklearn.linear_model import LogisticRegression

    clf_LogisticRegression = LogisticRegression(random_state=0, solver='lbfgs',
                                                multi_class='multinomial')

    # AdaBost
    from sklearn.ensemble import AdaBoostClassifier
    clf_AdaBoostClassifier = AdaBoostClassifier(
        n_estimators=100, random_state=0)

    classifiers.extend([
        [neigh, '1-NN Classifier'], [neigh_5, '5-NN Classifier'], [neigh_9,
                                                                   '9-NN Classifier'], [gnb, 'GaussianNB'],
        [DT_clf, 'DecisionTree '], [clf_LogisticRegression, 'Logistic Regression '], [clf_AdaBoostClassifier, 'AdaBoost ']])

    # classifiers.extend([ [neigh_5, '5-NN Classifier'], [neigh_9,'9-NN Classifier'], [gnb, 'GaussianNB'],
    #     [DT_clf, 'DecisionTree '], [clf_LogisticRegression, 'Logistic Regression '], [clf_AdaBoostClassifier, 'AdaBoost '], [clf_sgd, 'SGD']])

    acc_dict, n_fold, n_times = {}, 3, 3
    for _, title in classifiers:
        acc_dict[title] = []
    acc_dict['svm'] = []

    # Find best parameters for SVM classifier
    from sklearn import svm
    # lst_vals = [0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40]
    # best_acc, best_sigam, best_c, clf = 0, 0, 0, None
    # for sigam in lst_vals:
    #     for c_param in lst_vals:
    #         # train svm classifier by Gaussian kernel
    #         clf = svm.SVC(C=c_param, gamma=sigam, kernel='rbf',
    #                       decision_function_shape='ovo')
    #         y_pred = clf.fit(train_data, train_labels).predict(test_data)
    #         acc = accuracy_score(test_labels, y_pred)
    #         if acc > best_acc:
    #             best_acc, best_c, best_sigam = acc, c_param, sigam
    #             acc_dict['svm'].clear()
    #             acc_dict['svm'].append(acc)

    # 10 times 3 fold cv
    # for _ in range(n_times):
    #     samples, labels = shuffle(samples, train_labels+test_labels)
    #     for clf, title in classifiers:
    #         acc = np.mean(cross_val_score(clf, samples, labels, cv=n_fold))
    #         acc_dict[title].append(acc)

    # # 10 times 3 fold cv for SVM
    # for _ in range(n_times):
    #     samples, labels = shuffle(samples, train_labels+test_labels)
    #     clf = svm.SVC(C=best_c, gamma=best_sigam, kernel='rbf',
    #                   decision_function_shape='ovo')
    #     acc = np.mean(cross_val_score(clf, samples, labels, cv=n_fold))
    #     acc_dict[title].append(acc)

    # Evaluate models on given test dataset
    print(ds_title_preproc)
    for clf, title in classifiers:
        y_pred = clf.fit(train_data, train_labels).predict(test_data)
        plt.figure()
        plot_confusion_matrix(np.asarray(test_labels), y_pred, classes=class_names,
                              normalize=False, title=f'{title}')
        print(f'{title}: {accuracy_score(test_labels, y_pred)}')

    # clf = svm.SVC(C=best_c, gamma=best_sigam, kernel='rbf',
    #               decision_function_shape='ovo')
    clf = svm.SVC()
    y_pred = clf.fit(train_data, train_labels).predict(test_data)
    plt.figure()
    plot_confusion_matrix(np.asarray(test_labels), y_pred, classes=class_names,
                          normalize=False, title=f'SVM')
    print(f'SVM: {accuracy_score(test_labels, y_pred)}')
    # # print result
    # for key, val in acc_dict.items():
    #     print(f'{n_times} times {n_fold} fold CV: {key}: {round(np.mean(val),4)}')
    # print(f'SVM Parameter: C={best_c} , Sigma={best_sigam}')

    # plot_data(
    #     np.asarray(orginal_ds)[:, :2], labels,
    #     sample_icon='*',
    #     sample_color=['INDIANRED', 'DEEPPINK', 'MEDIUMVIOLETRED', 'DARKORANGE', 'GOLDENROD',
    #                   'MAGENTA', 'DARKSLATEBLUE', 'GREENYELLOW', 'STEELBLUE', 'DARKSLATEGRAY'],
    #     title='Orginal Dataset')
    plot_data(
        np.asarray(samples)[:, :2], labels,
        sample_icon='*',
        sample_color=['INDIANRED', 'DEEPPINK', 'MEDIUMVIOLETRED', 'DARKORANGE', 'GOLDENROD',
                      'MAGENTA', 'DARKSLATEBLUE', 'GREENYELLOW', 'STEELBLUE', 'DARKSLATEGRAY'],
        title=ds_title_preproc)

    plt.show()


main()
