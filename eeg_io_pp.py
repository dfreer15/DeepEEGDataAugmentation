import numpy as np
from mne.io import read_raw_edf, find_edf_events
import os
import pandas as pd
from numpy import genfromtxt
from scipy.signal import hilbert
import math
from time import clock


def get_data_2a(data_name, n_classes, num_channels=22):
    # # # Reads in raw edf (or gdf) file from BCI Competition 2a and returns the signal, time array,  
    # # # and start of each (labeled) motor imagery event
    # Returns: signal, time array, events
    
    freq = 250

    raw = read_raw_edf(data_name, preload=True, stim_channel='auto', verbose='WARNING')

    events = find_edf_events(raw)
    events.pop(0)
    time = events.pop(0)
    events1 = events.pop(0)
    events2 = events.pop(0)
    events3 = events.pop(0)

    # raw_train.plot_psd(area_mode='range', tmax=10.0)
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    signal = np.transpose(raw.get_data()[:num_channels])

    events = np.transpose(np.vstack([time, events1, events2, events3]))
    time = raw.times * freq

    return np.asarray(signal), time, events


def get_label_data_2a(data_name, n_classes, num_channels=22, remove_rest=True, training_data=True, reuse_data=False,
                mult_data=False, noise_data=False):

    freq = 250

    raw = read_raw_edf(data_name, preload=True, stim_channel='auto', verbose='WARNING')

    events = find_edf_events(raw)
    events.pop(0)
    time = events.pop(0)
    events1 = events.pop(0)
    events2 = events.pop(0)
    events3 = events.pop(0)

    # raw_train.plot_psd(area_mode='range', tmax=10.0)
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    signal = np.transpose(raw.get_data()[:num_channels])

    events = np.transpose(np.vstack([time, events1, events2, events3]))
    time = raw.times * freq

    if training_data:
        signal_out, labels = label_data_2a(signal, time, events, remove_rest, n_classes, freq, reuse_data=reuse_data,
                                           mult_data=mult_data, noise_data=noise_data)
        # if remove_rest:
        signal = signal_out
    else:
        labels = np.zeros(signal.shape[0])

    return np.asarray(signal), labels



def label_data_2a(signal, time, events, remove_rest, n_classes, freq):
    # # # Gets desired signal and matching labels for each time point
    # Returns: signal, labels 
    
    final_labels = []
    signal_out = np.zeros(signal.shape)
    t, s, j1 = 0, 0, 0
    cc_labels = 0

    min_event = 769
    if n_classes == 4 or n_classes == 5:
        max_event = 772
    elif n_classes == 3:
        max_event = 771
    elif n_classes == 2:
        if remove_rest:
            max_event = 770
        else:
            max_event = 772

    for j in range(len(time)):
        while events[t, 1] < min_event or events[t, 1] > max_event:
            t = t+1
            if t == len(events):
                signal_out = signal_out[:len(final_labels)]
                return signal_out, np.asarray(final_labels)

        if events[t, 0] + freq/2 < time[j] < events[t, 0] + freq * (5/2):
            if not remove_rest and n_classes == 2:
                final_labels.append(1)
                signal_out[j1] = signal[j]
                j1 += 1
            else:
                final_labels.append(events[t, 1] - 768)
                signal_out[j1] = signal[j]
                cc_labels += 1
                j1 += 1
        elif time[j] >= events[t, 0] + freq * (5/2):
            if not remove_rest and n_classes == 2:
                final_labels.append(1)
                signal_out[j1] = signal[j]
                j1 += 1
            else:
                final_labels.append(events[t, 1] - 768)
                signal_out[j1] = signal[j]
                cc_labels += 1
                j1 += 1
            t = t+1
        elif events[t, 0] < time[j] < events[t, 0] + freq/2:
            continue
        elif events[t, 0] + freq * (5/2) < time[j] < events[t, 0] + freq * 4:
            continue
        else:
            if remove_rest:
                continue
            elif s < cc_labels/4:
                final_labels.append(0)
                signal_out[j1] = signal[j]
                s += 1
                j1 += 1
            else:
                continue

        if t == len(events):
            signal_out = signal_out[:len(final_labels)]
            return signal_out, np.asarray(final_labels)

    signal_out = signal_out[:len(final_labels)]

    return np.asarray(signal_out), np.asarray(final_labels)


def label_data_2a_train(signal, time, events, freq, da_mod=1, reuse_data=False, mult_data=False, noise_data=False, neg_data=False, freq_mod_data=False):
    # # # Gets and labels desired signal
    # # # Counts how much rest data is required to maintain the 
    # # # proper balance (based on da_mod, and any data augmentation methods requested)
    # Returns: desired signal, corresponding labels
    
    final_labels = []
    signal_out = np.zeros(signal.shape)
    t, s, j1 = 0, 0, 0
    num_da_mod = 1
    if reuse_data:
        num_da_mod += 1
    if mult_data:
        num_da_mod += 2
    if noise_data:
        num_da_mod += 1
    if neg_data:
        num_da_mod += 1
    if freq_mod_data:
        num_da_mod += 2

    min_event = 769
    max_event = 772
    cc_labels = 0           # cc_labels counts how many times "control classes" (non resting classes) have been detected in the signal

    # 0 - rest; 1 - left; 2 - right; 3 - foot; 4 - tongue
    for j in range(len(time)):
        
        # This while loop ignores all events that are not motor imagery events
        # Motor imagery events have all been labelled with numbers 769-772
        while events[t, 1] < min_event or events[t, 1] > max_event:
            t = t+1
            # Formats and returns signal if all events have been considered
            if t == len(events):
                signal_out = signal_out[:len(final_labels)]
                num_to_add = check_train_set(final_labels, cc_labels, num_da_mod*da_mod)
                if num_to_add > 0:
                    np.concatenate([signal[first_j - num_to_add:first_j], signal_out])
                    np.concatenate([np.zeros(num_to_add), np.asarray(final_labels)])
                return signal_out, np.asarray(final_labels)

        # Adds MI signal/label only if it is between 0.5 and 2.5 seconds after the start of the event
        # This is done to remove unwanted Visually Evoked Potential (< 0.5s)
        # And to remove MI data that has "tapered off" in strength (> 2.5s)
        if events[t, 0] + freq/2 < time[j] < events[t, 0] + freq * (5/2):
            final_labels.append(events[t, 1] - 768)
            signal_out[j1] = signal[j]
            if j1 == 0:
                first_j = j
            j1 += 1
            cc_labels += 1
        # If the time is more than 4 seconds after the event began, the event number and time are incremented
        elif time[j] >= events[t, 0] + freq * 4:
            final_labels.append(events[t, 1] - 768)
            signal_out[j1] = signal[j]
            j1 += 1
            t += 1
        elif events[t, 0] < time[j] < events[t, 0] + freq/2:
            continue
        elif events[t, 0] + freq * (5/2) < time[j] < events[t, 0] + freq * 4:
            continue
        else:
            # "Resting" signal is added if it hasn't been overexpressed relative to the control classes
            # This is where counting cc_labels comes into play, and the data skew value (da_mod)
            if s < int(num_da_mod*da_mod*(cc_labels / 4)):
                final_labels.append(0)
                signal_out[j1] = signal[j]
                s += 1
                j1 += 1
            else:
                continue
    
        # Formats and returns signal if all events have been considered
        if t == len(events):
            signal_out = signal_out[:len(final_labels)]
            num_to_add = check_train_set(final_labels, cc_labels, num_da_mod*da_mod)
            if num_to_add > 0:
                np.concatenate([signal[first_j-num_to_add:first_j], signal_out])
                np.concatenate([np.zeros(num_to_add), np.asarray(final_labels)])
            return signal_out, np.asarray(final_labels)

    # Formats and returns signal
    signal_out = signal_out[:len(final_labels)]
    num_to_add = check_train_set(final_labels, cc_labels, num_da_mod*da_mod)
    if num_to_add > 0:
        np.concatenate([signal[first_j - num_to_add:first_j], signal_out])
        np.concatenate([np.zeros(num_to_add), np.asarray(final_labels)])

    return np.asarray(signal_out), np.asarray(final_labels)


def label_data_2a_val(signal, time, events, freq, remove_rest=False):
    # # # Gets entire signal and labels it, unless requested to remove "resting" data
    # # # Labels control class data differently from the above function: 0.5s-4s are all considered
    # # # as "control class" data. Data augmentation is not considered, as it isn't performed on test data
    # Returns: signal, labels
    
    final_labels = []
    t, s, j1 = 0, 0, 0

    if not remove_rest:
        signal_out = signal
    else:
        signal_out = np.zeros(signal.shape)

    min_event, max_event = 769, 772
    cc_labels = 0

    for j in range(len(time)):
        while events[t, 1] < min_event or events[t, 1] > max_event:
            t += 1
            if t == len(events):
                signal_out = signal_out[:len(final_labels)]
                return signal_out, np.asarray(final_labels)

        # if events[t, 0] + freq/2 < time[j] < events[t, 0] + freq * (5/2):
        if events[t, 0] + freq / 2 < time[j] < events[t, 0] + freq * 4:  # up to 4 seconds is labelled as the class
            final_labels.append(events[t, 1] - 768)
            cc_labels += 1
            if remove_rest:
                signal_out[j1] = signal[j]
                j1 += 1
        elif time[j] >= events[t, 0] + freq * 4:
            final_labels.append(events[t, 1] - 768)
            t += 1
            if remove_rest:
                signal_out[j1] = signal[j]
                j1 += 1
        elif remove_rest:
            if events[t, 0] < time[j] < events[t, 0] + freq / 2:
                continue
            elif events[t, 0] + freq * (5 / 2) < time[j] < events[t, 0] + freq * 4:
                continue
            elif s < int(cc_labels/4):
                signal_out[j1] = signal[j]
                final_labels.append(0)
                s += 1
                j1 += 1
            else:
                continue
        else:
            final_labels.append(0)

        if t == len(events):
            signal_out = signal_out[:len(final_labels)]
            return signal_out, np.asarray(final_labels)

    return signal_out, np.asarray(final_labels)


def process_data_2a(data, label, window_size, num_channels=22):
    # # # Takes as input a full signal with full labels
    # # # Segments both into windows, normalizes the signal, then formats and splits it
    # Returns: train_data, test_data, train_y, test_y as the segmented and normalized training and testing data and labels
    
    data, label = segment_signal_without_transition(data, label, window_size)
    unique, counts = np.unique(label, return_counts=True)
    data = norm_dataset(data)
    data = data.reshape([label.shape[0], window_size, num_channels])

    train_data, test_data, train_y, test_y = split_data(data, label)

    return train_data, test_data, train_y, test_y


def segment_signal_without_transition(data, label, window_size, overlap=1):
    # # # Divides signal and labels into segments of time. May be overlapping.
    # # # Ensures there is one label for each segment of window_size
    # Returns: segments, labels
    
    for (start, end) in windows(data, window_size, overlap=overlap):
        if len(data[start:end]) == window_size:
            x1_F = data[start:end]
            if start == 0:
                labels = label[end]
                segments = x1_F
            else:
                try:
                    labels = np.append(labels, label[end])
                    segments = np.vstack([segments, x1_F])
                except ValueError:
                    continue

    return segments, labels


def windows(data, size, overlap=1):
    # # # Returns the integer values that will serve as each beginning and ending indice
    # # # for the windows of data
    # Returns (yields): beginning, end for each window
    
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += (size / overlap)


def split_data(data_in_s, label_s, split_val=0.666):
    # # # Splits data and labels by a given split_val, with a split_val proportion of 
    # # # the data in the training set, and a (1 - split_val) proportion for the testing or validation set.
    # Returns: training and test sets, and labels
    
    split = int(split_val * len(label_s))
    train_x = data_in_s[0:split]
    train_y = label_s[0:split]

    test_x = data_in_s[split:]
    test_y = label_s[split:]

    return train_x, test_x, train_y, test_y


def norm_dataset(dataset):
    # # # Normailizes the entire dataset
    # Returns: normalised dataset
    norm_dataset = np.zeros(dataset.shape)
    for i in range(dataset.shape[0]):
        norm_dataset[i] = feature_normalize(dataset_1D[i])
    return norm_dataset


def feature_normalize(data):
    # # # Z-normalises one segment of data (timepoints, channels) by its entire mean and 
    # # # standard deviation
    # Returns: normalised data of the same shape as the input
    
    mean = data.mean()
    sigma = data.std()
    data_normalized = data
    data_normalized = (data_normalized - mean) / sigma
    data_normalized = (data_normalized - np.min(data_normalized))/np.ptp(data_normalized)

    return data_normalized


def data_aug(data, labels, size, reuse_data, mult_data, noise_data, neg_data, freq_mod_data):
    # # # Augments data based on boolean inputs reuse_data, noise_data, neg_data, freq_mod data.
    # Returns: entire training dataset after data augmentation, and the cooresponding labels
    
    n_channels = data.shape[2]
    data_out = data
    labels_out = labels

    if reuse_data:
        reuse_data_add, labels_reuse = data_reuse_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, reuse_data_add], axis=0)
        labels_out = np.append(labels_out, np.asarray(labels_reuse))
    if mult_data:
        mult_data_add, labels_mult = data_mult_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, mult_data_add], axis=0)
        labels_out = np.append(labels_out, np.asarray(labels_mult))
    if noise_data:
        noise_data_add, labels_noise = data_noise_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, noise_data_add], axis=0)
        labels_out = np.append(labels_out, np.asarray(labels_noise))
    if neg_data:
        neg_data_add, labels_neg = data_neg_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, neg_data_add], axis=0)
        labels_out = np.append(labels_out, np.asarray(labels_neg))
    if freq_mod_data:
        freq_data_add, labels_freq = freq_mod_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, freq_data_add], axis=0)
        labels_out = np.append(labels_out, np.asarray(labels_freq))

    return data_out, labels_out


def data_reuse_f(data, labels, size, n_channels=22):
    # Returns: data double the size of the input over time, repeating the input data
    
    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] > 0:
            new_data.append(data[i])
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])

    return new_data_ar, new_labels


def data_neg_f(data, labels, size, n_channels=22):
    # Returns: data double the size of the input over time, with new data
    # being a reflection along the amplitude 
    
    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] > 0:
            data_t = -1*data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])

    return new_data_ar, new_labels


def data_mult_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    mult_mod = 0.05
    print("mult mod: {}".format(mult_mod))
    for i in range(len(labels)):
        if labels[i] > 0:
            # print(data[i])
            data_t = data[i]*(1+mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] > 0:
            data_t = data[i]*(1-mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])

    return new_data_ar, new_labels


def data_noise_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    noise_mod_val = 2
    print("noise mod: {}".format(noise_mod_val))
    for i in range(len(labels)):
        if labels[i] > 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])

    return new_data_ar, new_labels


def freq_mod_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    print(data.shape)
    freq_mod = 0.2
    print("freq mod: {}".format(freq_mod))
    for i in range(len(labels)):
        if labels[i] > 0:
            low_shift = freq_shift(data[i], -freq_mod, num_channels=n_channels)
            new_data.append(low_shift)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] > 0:
            high_shift = freq_shift(data[i], freq_mod, num_channels=n_channels)
            new_data.append(high_shift)
            new_labels.append(labels[i])

    new_data_ar = np.asarray(new_data).reshape([-1, size, n_channels])

    return new_data_ar, new_labels

def freq_shift(x, f_shift, dt=1/250, num_channels=22):
    shifted_sig = np.zeros((x.shape))
    len_x = len(x)
    padding_len = 2**nextpow2(len_x)
    padding = np.zeros((padding_len - len_x, num_channels))
    with_padding = np.vstack((x, padding))
    hilb_T = hilbert(with_padding, axis=0)
    t = np.arange(0, padding_len)
    shift_func = np.exp(2j*np.pi*f_shift*dt*t)
    for i in range(num_channels):
        shifted_sig[:,i] = (hilb_T[:,i]*shift_func)[:len_x].real

    return shifted_sig

def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))

def check_train_set(final_labels, cc_labels, da_mod):
    num_rest_class = final_labels.count(0)
    print("da_mod: ", da_mod)
    if num_rest_class > int(cc_labels/4 * da_mod):
        return 0
    else:
        return int(cc_labels/4 * da_mod) - num_rest_class
