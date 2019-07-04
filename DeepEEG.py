import os.path

import eeg_io_pp
import pyriemann
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from torch import optim

import Deep_Func
import matplotlib.pyplot as plt


def get_clf(clf_method):
    if clf_method == "Riemann":
        fgda = pyriemann.tangentspace.FGDA()
        mdm = pyriemann.classification.MDM()
        clf = Pipeline([('FGDA', fgda), ('MDM', mdm)])
        return clf, False
    elif clf_method == "Braindecode":
        model = Deep_Func.create_model(n_classes, freq*window_size, in_chans=num_channels)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer
    elif clf_method == "LSTM":
        model = Deep_Func.create_model_lstm(n_classes, freq*window_size, in_chans=num_channels)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer


def transform_fit(clf, opt, train_data, train_labels, val_data, val_labels):
    if clf_method == "Riemann":
        cov_train = pyriemann.estimation.Covariances().fit_transform(np.transpose(train_data, axes=[0, 2, 1]))
        clf.fit_transform(cov_train, label_train)
        return clf, False
    elif clf_method == "Braindecode":
        # data = np.append(train_data, val_data, axis=0)
        # labels = np.append(train_labels, val_labels)

        train_data = (train_data * 1e6).astype(np.float32)
        val_data = (val_data*1e6).astype(np.float32)
        y = train_labels.astype(np.int64)
        X = np.transpose(train_data, [0, 2, 1])
        y_val = val_labels.astype(np.int64)
        print(val_data.shape)
        X_val = np.transpose(val_data, [0, 2, 1])
        model, accuracy_out = Deep_Func.fit_transform_2(clf, opt, X, y, X_val, y_val, input_time_length=int(freq * window_size),
                                            n_channels=num_channels, num_epochs=20)
        return model, accuracy_out
    elif clf_method == "LSTM":

        train_data = (train_data * 1e6).astype(np.float32)
        val_data = (val_data * 1e6).astype(np.float32)
        y = train_labels.astype(np.int64)
        y_val = val_labels.astype(np.int64)
        X = np.transpose(train_data, [0, 2, 1])
        X_val = np.transpose(val_data, [0, 2, 1])
        model, accuracy_out = Deep_Func.fit_transform_2(clf, opt, X, y, X_val, y_val, input_time_length=int(freq * window_size),
                                            n_channels=num_channels, num_epochs=20)
        return model, accuracy_out


def predict(clf, val_data, labels):
    if clf_method == "Riemann":
        # print("Val_data Shape: ", val_data.shape)
        cov_val = pyriemann.estimation.Covariances().fit_transform(np.transpose(val_data, axes=[0, 2, 1]))
        pred_val = clf.predict(cov_val)
        return pred_val
    elif clf_method == "Braindecode":
        val_data = (val_data * 1e6).astype(np.float32)
        X = np.transpose(val_data, [0, 2, 1])
        pred_val, cert = Deep_Func.predict(clf, X, labels, input_time_length=int(freq*window_size), n_channels=num_channels)
        return pred_val
    elif clf_method == "LSTM":
        val_data = (val_data * 1e6).astype(np.float32)
        X = np.transpose(val_data, [0, 2, 1])
        pred_val, cert = Deep_Func.predict(clf, X, labels, input_time_length=int(freq * window_size),
                                       n_channels=num_channels)
        return pred_val

if __name__ == '__main__':
    dataset = "bci_comp"
    # dataset = "gtec"

    running = True
    remove_rest = False
    reuse_data = False
    mult_data = False
    noise_data = False

    clf_methods = ["Riemann", "Braindecode", "LSTM"]
    # clf_methods = ["Braindecode", "LSTM"]
    subject_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # subject_nums = [2, 4, 5, 6]
    n_classes = 5
    # window_size_list = [2.5, 2, 1.5, 1, 0.5, 0.25]     # in seconds
    window_size_list = [2.5, 2, 1.5, 0.25]
    overlap_list = [1, 2]

    if dataset == "bci_comp":
        freq, num_channels = 250, 22
    elif dataset == "physionet":
        freq, num_channels = 160, 64
    elif dataset == "gtec":
        freq = 250
        num_channels = 32

    data_folder = '/data2/bci_data_preprocessed/'

    for clf_method in clf_methods:
        for window_size in window_size_list:
            for overlap in overlap_list:
                accuracy_f, precision_amb_f, precision_cc_f, recall_amb_f, recall_cc_f = [], [], [], [], []
                for subject_num in subject_nums:
                    # for n_classes in n_classes_list:
                    print("Classifier: " + clf_method)
                    print("Subject: {}".format(subject_num))
                    print("Window Size: {}".format(window_size))
                    print("Overlap: {}".format(overlap))
                    print("Classes: {}".format(n_classes))

                    data_file = dataset + '_sub' + str(subject_num) + '_' + str(n_classes) + 'class_window' + str(window_size) + '_overlap' + str(overlap)
                    if reuse_data:
                        data_file = data_file + '_reuse'
                    if mult_data:
                        data_file = data_file + '_mult'
                    if noise_data:
                        data_file = data_file + '_noise'

                    label_file = data_file + '_labels'

                    try:
                        data = np.load(data_folder + data_file + '.npy')
                        label = np.load(data_folder + label_file + '.npy')
                        print("Loading from datafile")
                    except FileNotFoundError:
                        print("Could not find datafile. Computing...")

                        if dataset == "bci_comp":
                            freq, num_channels = 250, 22
                            # dataset_dir = "/data2/bci_competition/BCICIV_2a_gdf/"
                            dataset_dir = "/data/users/df1215/MIdata/bci_competition/"
                            train_file = 'A' + format(subject_num, '02d') + 'T.gdf'
                            test_file = 'A' + format(subject_num, '02d') + 'E.gdf'
                            sig, sig_labels = eeg_io_pp.get_label_data_2a(dataset_dir + train_file, n_classes)
                        elif dataset == "physionet":
                            freq, num_channels = 160, 64
                            dataset_dir = "/data2/eegmmidb/"
                            sig, sig_labels = eeg_io_pp.get_data_PN(dataset_dir, subject_num, n_classes)
                            if num_channels == 16:
                                sig = eeg_io_pp.dataset_1Dto1D_16_PN(sig)
                        elif dataset == "gtec":
                            freq = 250
                            dataset_dir = "/data2/gtec_csv/"
                            file = "daniel_WET_3class"
                            sig, sig_labels = eeg_io_pp.get_data_gtec(dataset_dir, file, n_classes)

                        unique, counts = np.unique(sig_labels, return_counts=True)
                        print("Initial Labels: ", unique, counts)

                        data, label = eeg_io_pp.segment_signal_without_transition(sig, sig_labels, int(freq*window_size), overlap=overlap)

                        data = eeg_io_pp.norm_dataset(data)
                        data = data.reshape([label.shape[0], int(freq*window_size), num_channels])

                        print(data.shape)
                        print(label.shape)

                        # np.save(data_folder + data_file, data)
                        # np.save(data_folder + label_file, label)

                    train_len = int(0.667 * len(label))

                    data_train = data[:train_len]
                    data_val = data[train_len:]
                    label_train = label[:train_len]
                    label_val = label[train_len:]

                    unique, counts = np.unique(label_train, return_counts=True)
                    print("Training labels: ", unique, counts)
                    unique, counts = np.unique(label_val, return_counts=True)
                    print("Validation labels: ", unique, counts)

                    if reuse_data or mult_data or noise_data:
                        data_train, label_train = eeg_io_pp.data_aug(data_train, label_train, int(freq*window_size), reuse_data, mult_data, noise_data)

                    clf, opt = get_clf(clf_method)
                    if remove_rest:
                        label_train -= 1
                        label_val -= 1
                    clf, acc_out = transform_fit(clf, opt, data_train, label_train, data_val, label_val)
                    pred_val = predict(clf, data_val, label_val)

                    conf_mat = confusion_matrix(label_val, pred_val)
                    print(conf_mat)
                    tru_pos = []
                    for i in range(conf_mat.shape[0]):
                        tru_pos.append(conf_mat[i, i])

                    if subject_num == 1 or subject_num == 2:
                        conf_mat_f = conf_mat
                        acc_out_f = acc_out
                    else:
                        conf_mat_f = conf_mat_f + conf_mat
                        acc_out_f = acc_out_f + acc_out

                    tru_pos_1 = np.sum(conf_mat[1:, 1:])
                    false_pos_1 = np.sum(conf_mat[0, 1:])
                    false_neg_1 = np.sum(conf_mat[1:, 0])
                    # print(tru_pos_1, false_pos_1, false_neg_1)

                    print("# # # # # # # # # # # # # # # # # # # # # # #")
                    print(" ")
                    print("# # # # # # # # # # # # # # # # # # # # # # #")

                    print("Classifier: " + clf_method)
                    print("Window Size: {}".format(window_size))
                    print("Overlap: {}".format(overlap))
                    print("Classes: {}".format(n_classes))
                    print("Subject: {}".format(subject_num))

                    accuracy_val = np.sum(tru_pos) / (np.sum(conf_mat))
                    print("val accuracy: {}".format(accuracy_val))

                    specificity_1 = tru_pos_1/(tru_pos_1 + false_pos_1)
                    print("precision 1: {}".format(specificity_1))

                    sensitivity_1 = tru_pos_1/(tru_pos_1 + false_neg_1)
                    print("recall 1: {}".format(sensitivity_1))

                    accuracy_f.append(accuracy_val)
                    # precision_amb_f.append(precision_amb)
                    # precision_cc_f.append(precision_cc)
                    # recall_amb_f.append(recall_amb)
                    # recall_cc_f.append(recall_cc)

                    print("# # # # # # # # # # # # # # # # # # # # # # #")
                    print(" ")
                    print("# # # # # # # # # # # # # # # # # # # # # # #")

                print("Classifier: " + clf_method)
                print("Window Size: {}".format(window_size))
                print("Overlap: {}".format(overlap))
                print("Classes: {}".format(n_classes))

                print("Final Confusion Matrix: ")
                print(conf_mat_f)
                plt.plot(acc_out_f / 5)
                print("num_samples: {}".format(np.sum(conf_mat_f)))
                print("Overall Accuracy (stddev): {}  ({})".format(np.mean(accuracy_f), np.std(accuracy_f)))
                print("Ambient Precision (stddev): {}  ({})".format(np.mean(precision_amb_f), np.std(precision_amb_f)))
                print("Control Class Precision (stddev): {}  ({})".format(np.mean(precision_cc_f),
                                                                          np.std(precision_cc_f)))
                print("Ambient Recall (stddev): {}  ({})".format(np.mean(recall_amb_f), np.std(recall_amb_f)))
                print("Control Class Recall (stddev): {}  ({})".format(np.mean(recall_cc_f), np.std(recall_cc_f)))
                # plt.show()
                print("# # # # # # # # # # # # # # # # # # # # # # #")
                print(" ")
                print("# # # # # # # # # # # # # # # # # # # # # # #")

plt.show()