import mne
from mne.io import concatenate_raws, find_edf_events
from braindecode.mne_ext.signalproc import concatenate_raws_with_events

import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget

from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
# from ShallowFBCSP0 import ShallowFBCSPNet
from ShallowFBCSP_LSTM0 import ShallowFBCSPLSTM
# from SFBCSP_LSTM import ShallowFBCSPLSTM
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model
from torch import optim

from braindecode.torch_ext.util import np_to_var

from braindecode.datautil.iterators import CropsFromTrialsIterator

from braindecode.torch_ext.util import np_to_var, var_to_np
import torch.nn.functional as F
from numpy.random import RandomState
import torch as th
from braindecode.experiments.monitors import compute_preds_per_trial_from_crops

import eeg_io_pp

rng = RandomState((2017, 6, 30))
cuda = False


def create_model(n_classes, input_time_length, in_chans=22):

    set_random_seeds(seed=20170629, cuda=cuda)

    # This will determine how many crops are processed in parallel
    # final_conv_length determines the size of the receptive field of the ConvNet
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length,
                            final_conv_length=4, pool_time_length=20, pool_time_stride=5).create_network()
    to_dense_prediction_model(model)

    if cuda:
        model.cuda()

    return model


def create_model_lstm(n_classes, input_time_length, in_chans=22):

    set_random_seeds(seed=20170629, cuda=cuda)

    # This will determine how many crops are processed in parallel
    # final_conv_length determines the size of the receptive field of the ConvNet

    lstm_size = 30
    lstm_layers = 1

    print("#### LSTM SIZE {} ####".format(lstm_size))
    print("#### LSTM LAYERS {} ####".format(lstm_layers))

    model = ShallowFBCSPLSTM(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length, lstm_size=lstm_size, lstm_layers=lstm_layers,
                             n_filters_time=lstm_size, n_filters_spat=lstm_size, final_conv_length=4, pool_time_length=20, pool_time_stride=5).create_network()
    to_dense_prediction_model(model)

    if cuda:
        model.cuda()

    return model


def fit_transform(model, optimizer, data, labels, num_epochs=10, n_channels=22, input_time_length=500):
    # # # # # # # # CREATE CROPPED ITERATOR # # # # # # # # #

    train_set = SignalAndTarget(data, y=labels)

    # determine output size
    test_input = np_to_var(np.ones((2, n_channels, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    # print("{:d} predictions per input/trial".format(n_preds_per_input))

    iterator = CropsFromTrialsIterator(batch_size=32, input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    # # # # # # # # TRAINING LOOP # # # # # # #

    for i_epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        for batch_X, batch_y in iterator.get_batches(train_set, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            outputs = model(net_in)

            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss.backward()
            optimizer.step()

        model.eval()
        # print("Epoch {:d}".format(i_epoch))

        # Collect all predictions and losses
        all_preds = []
        all_losses = []
        batch_sizes = []
        for batch_X, batch_y in iterator.get_batches(train_set, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            outputs = model(net_in)
            all_preds.append(var_to_np(outputs))
            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss = float(var_to_np(loss))
            all_losses.append(loss)
            batch_sizes.append(len(batch_X))
        # Compute mean per-input loss
        loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                       np.mean(batch_sizes))
        # print("{:6s} Loss: {:.5f}".format('Train', loss))
        # Assign the predictions to the trials
        preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                          input_time_length,
                                                          train_set)
        # preds per trial are now trials x classes x timesteps/predictions
        # Now mean across timesteps for each trial to get per-trial predictions
        meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
        predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
        accuracy = np.mean(predicted_labels == train_set.y)
        # print("{:6s} Accuracy: {:.2f}%".format('Train', accuracy * 100))

    return model


def fit_transform_val(model, optimizer, X, y, num_epochs=20, n_channels=22, input_time_length=500):
    len_train = int(X.shape[0] * 0.667)

    train_set = SignalAndTarget(X[:len_train], y=y[:len_train])
    print(train_set.X.shape)
    test_set = SignalAndTarget(X[len_train:], y=y[len_train:])
    print(test_set.y.shape)

    # # # # # # # # CREATE CROPPED ITERATOR # # # # # # # # #

    # determine output size
    test_input = np_to_var(np.ones((2, n_channels, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    # print("{:d} predictions per input/trial".format(n_preds_per_input))

    iterator = CropsFromTrialsIterator(batch_size=32, input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    accuracy_out = []
    for i_epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        for batch_X, batch_y in iterator.get_batches(train_set, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            outputs = model(net_in)
            # Mean predictions across trial
            # Note that this will give identical gradients to computing
            # a per-prediction loss (at least for the combination of log softmax activation
            # and negative log likelihood loss which we are using here)
            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss.backward()
            optimizer.step()

        # Print some statistics each epoch
        model.eval()
        # print("Epoch {:d}".format(i_epoch))
        for setname, dataset in (('Train', train_set),('Test', test_set)):
            # Collect all predictions and losses
            all_preds = []
            all_losses = []
            batch_sizes = []
            for batch_X, batch_y in iterator.get_batches(dataset, shuffle=False):
                net_in = np_to_var(batch_X)
                if cuda:
                    net_in = net_in.cuda()
                net_target = np_to_var(batch_y)
                if cuda:
                    net_target = net_target.cuda()
                outputs = model(net_in)
                all_preds.append(var_to_np(outputs))
                outputs = th.mean(outputs, dim=2, keepdim=False)
                loss = F.nll_loss(outputs, net_target)
                loss = float(var_to_np(loss))
                all_losses.append(loss)
                batch_sizes.append(len(batch_X))
            # Compute mean per-input loss
            loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                           np.mean(batch_sizes))
            # print("{:6s} Loss: {:.5f}".format(setname, loss))
            # Assign the predictions to the trials
            # preds_per_trial = compute_preds_per_trial_from crops(all_preds,
            #                                                   input_time_length,
            #                                                   dataset)
            # preds per trial are now trials x classes x timesteps/predictions
            # Now mean across timesteps for each trial to get per-trial predictions
            meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
            predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
            accuracy = np.mean(predicted_labels == dataset.y)
            # print("{:6s} Accuracy: {:.2f}%".format(
            #     setname, accuracy * 100))
            if setname == 'Test':
                accuracy_out.append(accuracy)

    return model, np.asarray(accuracy_out)


def fit_transform_2(model, optimizer, train_data, y_train, test_data, y_test, num_epochs=20, n_channels=22, input_time_length=500):

    train_set = SignalAndTarget(train_data, y=y_train)
    test_set = SignalAndTarget(test_data, y=y_test)

    # # # # # # # # CREATE CROPPED ITERATOR # # # # # # # # #

    # determine output size
    test_input = np_to_var(np.ones((2, n_channels, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    iterator = CropsFromTrialsIterator(batch_size=32, input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    accuracy_out = []
    min_loss = 1000
    for i_epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        for batch_X, batch_y in iterator.get_batches(train_set, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            # print(batch_y)
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            outputs = model(net_in)
            # Mean predictions across trial
            # Note that this will give identical gradients to computing
            # a per-prediction loss (at least for the combination of log softmax activation
            # and negative log likelihood loss which we are using here)
            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss.backward()
            optimizer.step()

        # Print some statistics each epoch
        model.eval()
        # print("Epoch {:d}".format(i_epoch))
        for setname, dataset in (('Train', train_set),('Test', test_set)):
            # Collect all predictions and losses
            all_preds = []
            all_losses = []
            batch_sizes = []
            for batch_X, batch_y in iterator.get_batches(dataset, shuffle=False):
                net_in = np_to_var(batch_X)
                if cuda:
                    net_in = net_in.cuda()
                net_target = np_to_var(batch_y)
                if cuda:
                    net_target = net_target.cuda()
                outputs = model(net_in)
                all_preds.append(var_to_np(outputs))
                outputs = th.mean(outputs, dim=2, keepdim=False)
                loss = F.nll_loss(outputs, net_target)
                loss = float(var_to_np(loss))
                all_losses.append(loss)
                batch_sizes.append(len(batch_X))
            # Compute mean per-input loss
            loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                           np.mean(batch_sizes))
            # print("{:6s} Loss: {:.5f}".format(setname, loss))
            # Assign the predictions to the trials
            preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                              input_time_length,
                                                              dataset.X)
            # preds per trial are now trials x classes x timesteps/predictions
            # Now mean across timesteps for each trial to get per-trial predictions
            meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
            predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
            accuracy = np.mean(predicted_labels == dataset.y)
            # print("{:6s} Accuracy: {:.2f}%".format(setname, accuracy * 100))
            if setname == 'Test':
                accuracy_out.append(accuracy)
                if loss < min_loss:
                    min_loss = loss
                elif loss > min_loss * 1.1:
                    print("Training Stopping")
                    return model, np.asarray(accuracy_out)


    return model, np.asarray(accuracy_out)


def predict(model, data, labels, n_channels=22, input_time_length=500):

    # # # # # # # # CREATE CROPPED ITERATOR # # # # # # # # #
    val_set = SignalAndTarget(data, y=labels)

    # determine output size
    test_input = np_to_var(np.ones((2, n_channels, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    # print("{:d} predictions per input/trial".format(n_preds_per_input))

    iterator = CropsFromTrialsIterator(batch_size=32, input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    model.eval()

    # Collect all predictions and losses
    all_preds = []
    batch_sizes = []
    for batch_X, batch_y in iterator.get_batches(val_set, shuffle=False):
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        outputs = model(net_in)
        all_preds.append(var_to_np(outputs))
        outputs = th.mean(outputs, dim=2, keepdim=False)
        batch_sizes.append(len(batch_X))
    # Assign the predictions to the trials
    preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                      input_time_length,
                                                      val_set.X)
    # preds per trial are now trials x classes x timesteps/predictions
    # Now mean across timesteps for each trial to get per-trial predictions
    meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
    meaned_preds_per_trial_rec = -1/meaned_preds_per_trial
    label_cert = np.max(meaned_preds_per_trial_rec)
    predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
    accuracy = np.mean(predicted_labels == val_set.y)
    # print("{:6s} Accuracy: {:.2f}%".format('Validation', accuracy * 100))

    print('predicted_labels shape')
    print(predicted_labels.shape)

    return predicted_labels, label_cert


def predict_1(model, data, labels, n_channels=22, input_time_length=500):
    test_input = np_to_var(np.ones((2, n_channels, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    # print("{:d} predictions per input/trial".format(n_preds_per_input))

    net_in = np_to_var(data)
    if cuda:
        net_in = net_in.cuda()
    output = model(net_in)

    print(output)

    return output


def save_model(clf, model_file):
    th.save(clf.state_dict(), model_file)

    return


def load_model(model_file, in_chans=22, input_time_length=500):
    n_classes = 5
    model = ShallowFBCSPLSTM(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length, lstm_size=40,
                             final_conv_length=4, pool_time_length=20, pool_time_stride=5).create_network()
    # to_dense_prediction_model(model)
    model.load_state_dict(th.load(model_file))

    # model = th.load(model_file)

    return model


# # # # # # LOAD DATA # # # # # #
if __name__ == '__main__':

    n_classes = 4
    window_size = 500
    num_channels = 22

    for subject_num in range(1, 10):

        # dataset = 'physionet'
        dataset = 'bci_comp2a'

        print('# # # # # # # # # # # # # # # # # # #')
        print('Subject Number: ', subject_num)
        print('# # # # # # # # # # # # # # # # # # #')

        if dataset == 'bci_comp2a':
            dataset_dir = "/data2/bci_competition/BCICIV_2a_gdf/"
            file = 'A' + format(subject_num, '02d') + 'T.gdf'
            print(file)
            sig, sig_labels = eeg_io_pp.get_data_2a(dataset_dir + file, n_classes)
        elif dataset == 'physionet':
            dataset_dir = "/data2/eegmmidb/"
            sig, sig_labels = eeg_io_pp.get_data_PN(dataset_dir, subject_num, n_classes)
            # sig = eeg_io_pp.dataset_1Dto1D_16_PN(sig)

        data, label = eeg_io_pp.segment_signal_without_transition(sig, sig_labels, window_size)
        data = eeg_io_pp.norm_dataset(data)
        data = data.reshape([label.shape[0], window_size, num_channels])

        # # # # # # # # CONVERT TO BDC FORMAT # # # # # # # # #

        # Convert data from volt to millivolt
        # Pytorch expects float32 for input and int64 for labels.
        X = (data * 1e6).astype(np.float32)
        y = (label - 1).astype(np.int64)

        X = np.transpose(X, [0, 2, 1])

        len_train = int(X.shape[0]*0.667)

        train_set = SignalAndTarget(X[:len_train], y=y[:len_train])
        print(train_set.X.shape)
        test_set = SignalAndTarget(X[len_train:], y=y[len_train:])
        print(test_set.y.shape)

        model = create_model(n_classes, window_size)
        optimizer = optim.Adam(model.parameters())
        fit_transform(model, optimizer, train_set.X, train_set.y)


