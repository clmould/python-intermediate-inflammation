"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import json
import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=",")


def load_json(filename):
    """Load a numpy array from a JSON document.

    Expected format:
    [
        {
            observations: [0, 1]
        },
        {
            observations: [0, 2]
        }
    ]

    :param filename: Filename of CSV to load

    """
    with open(filename, "r", encoding="utf-8") as file:
        data_as_json = json.load(file)
        return [np.array(entry["observations"]) for entry in data_as_json]


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array for each day.

    :param data: A 2D data array with inflammation data (each row contains measurements
    for a single patient across all days).
    :returns: An array of mean values of measurements for each day.
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array for each day.

    :param data: A 2D data array with inflammation data (each row contains measurements
    for a single patient across all days).
    :returns: An array of max values of measurements for each day.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array for each day.

    :param data: A 2D data array with inflammation data (each row contains measurements
    for a single patient across all days).
    :returns: An array of minimum values of measurements for each day.
    """
    return np.min(data, axis=0)


def s_dev(data):
    """Computes and returns standard deviation for data."""
    mean_data = daily_mean(data)
    devs = []
    for entry in data:
        devs.append((entry - mean_data) * (entry - mean_data))

    s_dev2 = sum(devs) / len(data)
    return {'standard deviation': s_dev2}

def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data input should be ndarray")
    if len(data.shape) != 2:
        raise ValueError("inflammation array should be 2-dimensional")
    if np.any(data < 0):
        raise ValueError("Inflammation values should not be negative")
    maxima = np.nanmax(data, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = data / maxima[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised
