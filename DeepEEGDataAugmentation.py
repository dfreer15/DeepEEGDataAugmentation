import os.path

import eeg_io_pp
import pyriemann
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from torch import optim

import Deep_Func
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_clf(clf_method):
    # Set classification method and return the classifier/model and optimizer, if applicable
    
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


def transform_fit(clf, opt, train_data, train_labels, test_data, test_labels):
    # Train the classifier based using training data and labels
    # Test the model using test data
    # Returns the trained model and the accuracy
    
    if clf_method == "Riemann":
        cov_train = pyriemann.estimation.Covariances().fit_transform(np.transpose(train_data, axes=[0, 2, 1]))
        clf.fit(cov_train, train_labels)
        return clf, False
    elif clf_method == "Braindecode":
        train_data = (train_data * 1e6).astype(np.float32)
        test_data = (test_data * 1e6).astype(np.float32)
        y = train_labels.astype(np.int64)
        X = np.transpose(train_data, [0, 2, 1])
        X_test = np.transpose(test_data, [0, 2, 1])
        y_test = test_labels.astype(np.int64)

        model, accuracy_out = Deep_Func.fit_transform_2(clf, opt, X, y, X_test, y_test, input_time_length=int(freq * window_size),
                                            n_channels=num_channels, num_epochs=50)
        return model, accuracy_out
    elif clf_method == "LSTM":
        train_data = (train_data * 1e6).astype(np.float32)
        test_data = (test_data * 1e6).astype(np.float32)
        X = np.transpose(train_data, [0, 2, 1])
        y = train_labels.astype(np.int64)
        X_test = np.transpose(test_data, [0, 2, 1])
        y_test = test_labels.astype(np.int64)
        model, accuracy_out = Deep_Func.fit_transform_2(clf, opt, X, y, X_test, y_test, input_time_length=int(freq * window_size),
                                            n_channels=num_channels, num_epochs=50)
        return model, accuracy_out


def predict(clf, val_data, labels):
    # Trained classifier makes a prediction on incoming values, and returns the prediction
    
    if clf_method == "Riemann":
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
    # dataset = "physionet"

    running = True
    final_testing = True
    n_classes = 5
    if n_classes == 5:
        remove_all_rest = False
    else:
        remove_all_rest = True
    remove_rest_val = False
    reuse_data = False
    mult_data = False
    noise_data = False
    neg_data = False
    freq_mod_data = False

    clf_methods = ["Riemann", "Braindecode", "LSTM"]
    # clf_methods = ["Riemann"]
    subject_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # subject_nums = [7, 8]
    # for j in range(1, 10):
    #     subject_nums.append(j)
    # window_size_list = [0.25, 0.5, 1, 1.5, 2]     # in seconds
    window_size_list = [0.5]
    overlap_list = [3]
    # overlap_list = [1, 2, 3]
    # da_mods = [1, 1.25, 1.5, 2, 2.5]
    da_mods = [3, 3.5, 4, 4.5, 5]

    if dataset == "bci_comp":
        freq, num_channels = 250, 22
    elif dataset == "physionet":
        freq, num_channels = 160, 64
    elif dataset == "gtec":
        freq = 250
        num_channels = 32

    # data_folder = '/data/users/df1215/MIdata/'  # On fuhe
    data_folder = '/data2/bci_data_preprocessed/' # elasa

    accuracy_out, precision_amb_out, precision_cc_out, recall_amb_out, recall_cc_out = [], [], [], [], []
    bal_acc_out, bal_acc_r_out = [], []
    accuracy_out_test, precision_amb_out_test, precision_cc_out_test, recall_amb_out_test = [], [], [], []
    recall_cc_out_test, bal_acc_out_test, bal_acc_r_out_test = [], [], []
    runs = [4]
    # runs = [0, 1, 2, 3, 4, 5, 6, 7]
    for clf_method in clf_methods:
        for run in runs:
            if run == 1:
                noise_data = True
            elif run == 2:
                neg_data = True
                noise_data = False
            elif run == 3:
                noise_data = False
                neg_data = False
                mult_data = True
            elif run == 4:
                mult_data = False
                freq_mod_data = True
            elif run == 5:
                freq_mod_data = False
                neg_data = True
                noise_data = True
            elif run == 6:
                noise_data = True
                neg_data = False
                mult_data = True
            elif run == 7:
                mult_data, noise_data = False, False
                neg_data, freq_mod_data = True, True
            for overlap in overlap_list:
                for window_size in window_size_list:
                    for da_mod in da_mods:
                        accuracy_f, bal_acc_f, precision_amb_f, precision_cc_f, recall_amb_f, recall_cc_f = [],[],[],[],[], []
                        for subject_num in subject_nums:
                            # for n_classes in n_classes_list:
                            print("Classifier: " + clf_method)
                            print("Data Aug Mod: {}".format(da_mod))
                            print("Window Size: {}".format(window_size))
                            print("Overlap: {}".format(overlap))
                            # print("Classes: {}".format(n_classes))
                            print("Subject: {}".format(subject_num))
                            print("Mult: {}".format(mult_data))
                            print("Noise: {}".format(noise_data))
                            print("Neg: {}".format(neg_data))
                            print("Freq: {}".format(freq_mod_data))

                            data_file = dataset + '_sub' + str(subject_num) + '_window' + str(window_size) + '_overlap' + str(overlap) + '_damod' + str(da_mod)
                            data_file = data_file + '_' + str(n_classes) + 'class'
                            if not remove_rest_val:
                                data_file = data_file + '_allval'
                            if reuse_data:
                                data_file = data_file + '_reuse'
                            if mult_data:
                                data_file = data_file + '_mult'
                            if noise_data:
                                data_file = data_file + '_noise'
                            if neg_data:
                                data_file = data_file + '_neg'

                            label_file = data_file + '_labels'

                            try:
                                data_train = np.load(data_folder + data_file + '_train.npy')
                                label_train = np.load(data_folder + label_file + '_train.npy')
                                data_val = np.load(data_folder + data_file + '_val.npy')
                                label_val = np.load(data_folder + label_file + '_val.npy')
                                print("Loading from datafile")
                            except FileNotFoundError:
                                print("Could not find datafile. Computing...")

                                if dataset == "bci_comp":
                                    freq, num_channels = 250, 22
                                    dataset_dir = "/data2/bci_competition/BCICIV_2a_gdf/"
                                    # dataset_dir = "/data/users/df1215/MIdata/bci_competition/"
                                    train_file = 'A' + format(subject_num, '02d') + 'T.gdf'
                                    test_file = 'A' + format(subject_num, '02d') + 'E.gdf'
                                    sig, time, events = eeg_io_pp.get_data_2a(dataset_dir + train_file, n_classes)
                                    train_len = int(0.6 * len(sig))
                                    val_len = int(0.8 * len(sig))
                                    print('Train Len: ', train_len)
                                    sig_train = sig[:train_len]
                                    time_train = time[:train_len]
                                    sig_val = sig[train_len:val_len]
                                    time_val = time[train_len:val_len]
                                    if final_testing:
                                        sig_test = sig[val_len:]
                                        time_test = time[val_len:]

                                    if remove_all_rest:
                                        signal_train, labels_train = eeg_io_pp.label_data_2a(sig_train, time_train, events, True, n_classes, freq)
                                        signal_val, labels_val = eeg_io_pp.label_data_2a(sig_val, time_val, events, True, n_classes, freq)
                                    else:
                                        print('Train: ')
                                        signal_train, labels_train = eeg_io_pp.label_data_2a_train(sig_train, time_train, events, freq,
                                                                                                     da_mod=da_mod, reuse_data=reuse_data,
                                                                                                     mult_data=mult_data, noise_data=noise_data,
                                                                                                     neg_data=neg_data, freq_mod_data=freq_mod_data)
                                        print('Validation: ')
                                        signal_val, labels_val = eeg_io_pp.label_data_2a_val(sig_val, time_val, events, freq, remove_rest=remove_rest_val)
                                        if signal_val.shape[0] != len(labels_val):
                                            signal_val = signal_val[:len(labels_val)]
                                        print('Sig Out: ', signal_train.shape)

                                unique, counts = np.unique(labels_train, return_counts=True)
                                print("Initial Labels (train): ", unique, counts)
                                unique, counts = np.unique(labels_val, return_counts=True)
                                print("Initial Labels (val): ", unique, counts)

                                data_train, label_train = eeg_io_pp.segment_signal_without_transition(signal_train, labels_train, int(freq*window_size), overlap=overlap)
                                data_val, label_val = eeg_io_pp.segment_signal_without_transition(signal_val, labels_val, int(freq*window_size), overlap=overlap)

                                unique, counts = np.unique(label_train, return_counts=True)
                                print("Labels after Segmentation (train): ", unique, counts)
                                unique, counts = np.unique(label_val, return_counts=True)
                                print("Labels after Segmentation (val): ", unique, counts)

                                data_train = eeg_io_pp.norm_dataset(data_train)
                                data_val = eeg_io_pp.norm_dataset(data_val)
                                data_train = data_train.reshape([label_train.shape[0], int(freq*window_size), num_channels])
                                data_val = data_val.reshape([label_val.shape[0], int(freq * window_size), num_channels])

                                if reuse_data or mult_data or noise_data or neg_data or freq_mod_data:
                                    data_train, label_train = eeg_io_pp.data_aug(data_train, label_train, int(freq * window_size), reuse_data, mult_data, noise_data, neg_data, freq_mod_data)

                                # np.save(data_folder + data_file + '_train', data_train)
                                # np.save(data_folder + data_file + '_val', data_val)
                                # np.save(data_folder + label_file + '_train', label_train)
                                # np.save(data_folder + label_file + '_val', label_val)

                            unique, counts = np.unique(label_train, return_counts=True)
                            print("Training labels: ", unique, counts)
                            unique, counts = np.unique(label_val, return_counts=True)
                            print("Validation labels: ", unique, counts)

                            clf, opt = get_clf(clf_method)
                            if remove_all_rest:
                                label_train -= 1
                                label_val -= 1
                            clf, acc_out = transform_fit(clf, opt, data_train, label_train, data_val, label_val)
                            pred_val = predict(clf, data_val, label_val)

                            conf_mat = confusion_matrix(label_val, pred_val)

                            if subject_num == 1:
                                conf_mat_f = conf_mat
                            else:
                                conf_mat_f = conf_mat_f + conf_mat

                            print("# # # # # # # # # # # # # # # # # # # # # # #")
                            print(" ")
                            print("# # # # # # # # # # # # # # # # # # # # # # #")

                            if final_testing:
                                if dataset == "bci_comp":
                                    dataset_dir = "/data2/bci_competition/BCICIV_2a_gdf/"
                                    # dataset_dir = "/data/users/df1215/MIdata/bci_competition/"
                                    train_file = 'A' + format(subject_num, '02d') + 'T.gdf'

                                    sig, time, events = eeg_io_pp.get_data_2a(dataset_dir + train_file, n_classes)

                                    val_len = int(0.8 * len(sig))

                                    sig_test = sig[val_len:]
                                    time_test = time[val_len:]

                                    if remove_all_rest:
                                        signal_test, labels_test = eeg_io_pp.label_data_2a(sig_test, time_test, events,
                                                                                       True, n_classes, freq)
                                    else:
                                        signal_test, labels_test = eeg_io_pp.label_data_2a_val(sig_test, time_test,
                                                                                               events, freq,
                                                                                               remove_rest=remove_rest_val)

                                    if signal_test.shape[0] != len(labels_test):
                                        signal_test = signal_test[:len(labels_test)]

                                data_test, label_test = eeg_io_pp.segment_signal_without_transition(signal_test,
                                                                                                    labels_test, int(
                                        freq * window_size), overlap=overlap)
                                unique, counts = np.unique(label_test, return_counts=True)
                                print("Labels after Segmentation (test): ", unique, counts)

                                data_test = eeg_io_pp.norm_dataset(data_test)
                                data_test = data_test.reshape(
                                    [label_test.shape[0], int(freq * window_size), num_channels])

                                if remove_all_rest:
                                    label_test -= 1

                                unique, counts = np.unique(label_test, return_counts=True)
                                print("Test labels: ", unique, counts)

                                pred_test = predict(clf, data_test, label_test)
                                conf_mat_test = confusion_matrix(label_test, pred_test)
                                conf_mat = conf_mat_test

                                if subject_num == 1:
                                    conf_mat_f_test = conf_mat_test
                                    # acc_out_f_test = acc_out
                                else:
                                    conf_mat_f_test = conf_mat_f_test + conf_mat_test
                                    # acc_out_f_test = acc_out_f_test + acc_out


                            print("Classifier: " + clf_method)
                            print("Data Aug Mod: {}".format(da_mod))
                            print("Window Size: {}".format(window_size))
                            print("Overlap: {}".format(overlap))
                            print("Subject: {}".format(subject_num))
                            print("Mult: {}".format(mult_data))
                            print("Noise: {}".format(noise_data))
                            print("Neg: {}".format(neg_data))
                            print("Freq: {}".format(freq_mod_data))

                            print(conf_mat)

                            tru_pos, prec_i, recall_i = [], [], []
                            for i in range(conf_mat.shape[0]):
                                tru_pos.append(conf_mat[i, i])
                                prec_i.append(conf_mat[i, i] / np.sum(conf_mat[:, i]).astype(float))
                                recall_i.append(conf_mat[i, i] / np.sum(conf_mat[i, :]).astype(float))

                            print("num_samples: {}".format(np.sum(conf_mat)))

                            accuracy_val = np.sum(tru_pos).astype(float) / (np.sum(conf_mat)).astype(float)
                            print("accuracy: {}".format(accuracy_val))

                            precision_amb = prec_i[0]
                            print("ambient precision: {}".format(precision_amb))

                            precision_cc = np.sum(prec_i[1:]) / (conf_mat.shape[0] - 1)
                            print("control class precision: {}".format(precision_cc))

                            recall_amb = recall_i[0]
                            print("ambient recall: {}".format(recall_amb))

                            recall_cc = np.sum(recall_i[1:]) / (conf_mat.shape[0] - 1)
                            print("control class recall: {}".format(recall_cc))

                            bal_acc = np.sum(prec_i)/conf_mat.shape[0]
                            print("balanced accuracy: {}".format(bal_acc))

                            bal_acc_r = np.sum(recall_i) / conf_mat.shape[0]
                            print("balanced accuracy (recall): {}".format(bal_acc_r))

                            accuracy_f.append(accuracy_val)
                            bal_acc_f.append(bal_acc)
                            precision_amb_f.append(precision_amb)
                            precision_cc_f.append(precision_cc)
                            recall_amb_f.append(recall_amb)
                            recall_cc_f.append(recall_cc)

                            accuracy_out.append(accuracy_val)
                            precision_amb_out.append(precision_amb)
                            precision_cc_out.append(precision_cc)
                            recall_amb_out.append(recall_amb)
                            recall_cc_out.append(recall_cc)
                            bal_acc_out.append(bal_acc)
                            bal_acc_r_out.append(bal_acc_r)


                            print("# # # # # # # # # # # # # # # # # # # # # # #")
                            print(" ")
                            print("# # # # # # # # # # # # # # # # # # # # # # #")

                        print("Classifier: " + clf_method)
                        print("Data Aug Mod: {}".format(da_mod))
                        print("Window Size: {}".format(window_size))
                        print("Overlap: {}".format(overlap))
                        # print("Classes: {}".format(n_classes))
                        print("Mult: {}".format(mult_data))
                        print("Noise: {}".format(noise_data))
                        print("Neg: {}".format(neg_data))
                        print("Freq: {}".format(freq_mod_data))

                        print("Final Confusion Matrix: ")
                        print(conf_mat_f)
                        # plt.plot(acc_out_f/5)

                        print("num_samples: {}".format(np.sum(conf_mat_f)))
                        print("Overall Accuracy (stddev): {}  ({})".format(np.mean(accuracy_f), np.std(accuracy_f)))
                        print("Balanced Accuracy (stddev): {} ({})".format(np.mean(bal_acc_f), np.std(bal_acc_f)))
                        print("Ambient Precision (stddev): {}  ({})".format(np.mean(precision_amb_f), np.std(precision_amb_f)))
                        print("Control Class Precision (stddev): {}  ({})".format(np.mean(precision_cc_f), np.std(precision_cc_f)))
                        print("Ambient Recall (stddev): {}  ({})".format(np.mean(recall_amb_f), np.std(recall_amb_f)))
                        print("Control Class Recall (stddev): {}  ({})".format(np.mean(recall_cc_f), np.std(recall_cc_f)))
                        # plt.show()
                        print("# # # # # # # # # # # # # # # # # # # # # # #")
                        print(" ")
                        print("# # # # # # # # # # # # # # # # # # # # # # #")

                        if final_testing:
                            print("# # # # # # # # # # # # # # # # # # # # # # #")
                            print(" ")
                            print("# # # # # # # # # # # # # # # # # # # # # # #")

                            print("Classifier: " + clf_method)
                            print("Data Aug Mod: {}".format(da_mod))
                            print("Window Size: {}".format(window_size))
                            print("Overlap: {}".format(overlap))
                            # print("Classes: {}".format(n_classes))
                            print("Mult: {}".format(mult_data))
                            print("Noise: {}".format(noise_data))
                            print("Neg: {}".format(neg_data))
                            print("Freq: {}".format(freq_mod_data))

                            print("Final Test Confusion Matrix: ")
                            print(conf_mat_f_test)

                            tru_pos, prec_i, recall_i = [], [], []
                            for i in range(conf_mat_f_test.shape[0]):
                                tru_pos.append(conf_mat_f_test[i, i])
                                prec_i.append(conf_mat_f_test[i, i] / np.sum(conf_mat_f_test[:, i]).astype(float))
                                recall_i.append(conf_mat_f_test[i, i] / np.sum(conf_mat_f_test[i, :]).astype(float))

                            accuracy_test = np.sum(tru_pos).astype(float) / (np.sum(conf_mat_f_test)).astype(float)
                            bal_acc = np.sum(prec_i) / conf_mat_f_test.shape[0]
                            bal_acc_r = np.sum(recall_i) / conf_mat_f_test.shape[0]

                            print("num_samples: {}".format(np.sum(conf_mat_f_test)))

                            print("Overall Accuracy: {}".format(accuracy_test))
                            print("Ambient Precision: {}".format(prec_i[0]))
                            print("Control Class Precision: {}".format(np.mean(prec_i[1:])))
                            print("Ambient Recall: {}".format(np.mean(recall_i[0])))
                            print("Control Class Recall: {}".format(np.mean(recall_i[1:])))
                            print("Balanced Accuracy (prec): {}".format(bal_acc))
                            print("Balanced Accuracy (rec): {}".format(bal_acc_r))

                            accuracy_out_test.append(accuracy_test)
                            precision_amb_out_test.append(prec_i[0])
                            precision_cc_out_test.append(np.mean(prec_i[1:]))
                            recall_amb_out_test.append(recall_i[0])
                            recall_cc_out_test.append(np.mean(recall_i[1:]))
                            bal_acc_out_test.append(bal_acc)
                            bal_acc_r_out_test.append(bal_acc_r)

                            # plt.show()
                            print("# # # # # # # # # # # # # # # # # # # # # # #")
                            print(" ")
                            print("# # # # # # # # # # # # # # # # # # # # # # #")


    accuracy_out_f = np.asarray(accuracy_out).reshape([len(runs)*len(clf_methods), len(da_mods), len(subject_nums)])
    precision_amb_out_f = np.asarray(precision_amb_out).reshape([len(runs)*len(clf_methods), len(da_mods), len(subject_nums)])
    precision_cc_out_f = np.asarray(precision_cc_out).reshape([len(runs)*len(clf_methods), len(da_mods), len(subject_nums)])
    recall_amb_out_f = np.asarray(recall_amb_out).reshape([len(runs)*len(clf_methods), len(da_mods), len(subject_nums)])
    recall_cc_out_f = np.asarray(recall_cc_out).reshape([len(runs)*len(clf_methods), len(da_mods), len(subject_nums)])
    bal_acc_out_f = np.asarray(bal_acc_out).reshape([len(runs)*len(clf_methods), len(da_mods), len(subject_nums)])
    bal_acc_r_out_f = np.asarray(bal_acc_r_out).reshape([len(runs) * len(clf_methods), len(da_mods), len(subject_nums)])

    accuracy_out_fin = np.transpose(accuracy_out_f, [2, 0, 1])
    precision_amb_out_fin = np.transpose(precision_amb_out_f, [2, 0, 1])
    precision_cc_out_fin = np.transpose(precision_cc_out_f, [2, 0, 1])
    recall_amb_out_fin = np.transpose(recall_amb_out_f, [2, 0, 1])
    recall_cc_out_fin = np.transpose(recall_cc_out_f, [2, 0, 1])
    bal_acc_out_fin = np.transpose(bal_acc_out_f, [2, 0, 1])
    bal_acc_r_out_fin = np.transpose(bal_acc_r_out_f, [2, 0, 1])

    accuracy_out_f_test = np.asarray(accuracy_out_test).reshape([len(runs)*len(clf_methods), len(da_mods)])
    preca_out_f_test = np.asarray(precision_amb_out_test).reshape([len(runs)*len(clf_methods), len(da_mods)])
    precc_out_f_test = np.asarray(precision_cc_out_test).reshape([len(runs)*len(clf_methods), len(da_mods)])
    reca_out_f_test = np.asarray(recall_amb_out_test).reshape([len(runs)*len(clf_methods), len(da_mods)])
    recc_out_f_test = np.asarray(recall_cc_out_test).reshape([len(runs)*len(clf_methods), len(da_mods)])
    bal_acc_out_f_test = np.asarray(bal_acc_out_test).reshape([len(runs)*len(clf_methods), len(da_mods)])
    bal_acc_r_out_f_test = np.asarray(bal_acc_r_out_test).reshape([len(runs)*len(clf_methods), len(da_mods)])


    sjt = 0

    for subject in subject_nums:
        np.savetxt('acc_subject' + str(subject) + '_clstm.csv', accuracy_out_fin[sjt], delimiter=',')
        np.savetxt('pra_subject' + str(subject) + '_clstm.csv', precision_amb_out_fin[sjt], delimiter=',')
        np.savetxt('prc_subject' + str(subject) + '_clstm.csv', precision_cc_out_fin[sjt], delimiter=',')
        np.savetxt('reca_subject' + str(subject) + '_clstm.csv', recall_amb_out_fin[sjt], delimiter=',')
        np.savetxt('recc_subject' + str(subject) + '_clstm.csv', recall_cc_out_fin[sjt], delimiter=',')
        np.savetxt('bap_subject' + str(subject) + '_clstm.csv', bal_acc_out_fin[sjt], delimiter=',')
        np.savetxt('bar_subject' + str(subject) + '_clstm.csv', bal_acc_r_out_fin[sjt], delimiter=',')
        sjt = sjt+1

    np.savetxt('acc_all_clstm.csv', accuracy_out_f_test, delimiter=',')
    np.savetxt('pra_all_clstm.csv', preca_out_f_test, delimiter=',')
    np.savetxt('prc_all_clstm.csv', precc_out_f_test, delimiter=',')
    np.savetxt('reca_all_clstm.csv', reca_out_f_test, delimiter=',')
    np.savetxt('recc_all_clstm.csv', recc_out_f_test, delimiter=',')
    np.savetxt('bap_all_clstm.csv', bal_acc_out_f_test, delimiter=',')
    np.savetxt('bar_all_clstm.csv', bal_acc_r_out_f_test, delimiter=',')

    # plt.show()
