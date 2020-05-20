"""
Kei Imada
20170801
A Butterworth signal filter worth using
"""

__all__ = ["Butter"]
__version__ = "1.0"
__author__ = "Kei Imada"


import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _filterHelper(x, w, f, N):
    """
    x a float
    w an array of arrays of floats
    f an array of arrays of floats
    N an int
    """
    w[0][4] = x
    for m in range(N / 2):
        previous_x = w[m]
        previous_y = w[m + 1]

        ym = f[0][m] * (
            previous_x[4]
            + f[1][m] * previous_x[3]
            + f[2][m] * previous_x[2]
            + f[3][m] * previous_x[1]
            + f[4][m] * previous_x[0]
        ) - (
            f[5][m] * previous_y[3]
            + f[6][m] * previous_y[2]
            + f[7][m] * previous_y[1]
            + f[8][m] * previous_y[0]
        )

        previous_y[4] = ym

        for i in range(len(previous_x) - 1):
            previous_x[i] = previous_x[i + 1]
    for i in range(len(previous_y) - 1):
        previous_y[i] = previous_y[i + 1]
    return ym


class Butter(object):
    def __init__(self, btype="lowpass", cutoff=None,
                 cutoff1=None, cutoff2=None,
                 rolloff=48, sampling=None):
        """The constructor for the butter filter object
        @param btype string type of filter, default lowpass
            lowpass
            highpass
            bandpass
            notch
            bandstop
        filter required arguments
            @param rolloff float measured in dB/Oct, default 48Hz
            @param sampling float measured in Hz
        lowpass filter required arguments
            @param cutoff float measured in Hz
        highpass filter required arguments
            @param cutoff float measured in Hz
        bandpass filter required arguments
            @param cutoff1 float measured in Hz
            @param cutoff2 float measured in Hz
            cutoff1 < cutoff2
        notch filter required arguments
            @param cutoff float measured in Hz
        bandstop filter required arguments
            @param cutoff1 float measured in Hz
            @param cutoff2 float measured in Hz
            cutoff1 < cutoff2
        """
        # input checking
        valid = map(lambda k: k[0],
                    filter(lambda k: type(k[1]) in [int, float],
                           zip(["cutoff", "cutoff1", "cutoff2", "rolloff", "sampling"],
                               [cutoff, cutoff1, cutoff2, rolloff, sampling])
                           )
                    )
        valid = list(valid)
        if None in [rolloff, sampling]:
            raise ValueError(
                "Butter:rolloff and sampling required for %s filter" % btype)
        if "rolloff" not in valid:
            raise TypeError("Butter:invalid rolloff argument")
        if "sampling" not in valid:
            raise TypeError("Butter:invalid sampling argument")
        if btype in ["lowpass", "highpass", "notch"]:
            if None in [cutoff]:
                raise ValueError(
                    "Butter:cutoff required for %s filter" % btype)
            if "cutoff" not in valid:
                raise TypeError("Butter:invalid cutoff argument")
        elif btype in ["bandpass", "bandstop"]:
            if None in [cutoff1, cutoff2]:
                raise ValueError(
                    "Butter:cutoff1 and cutoff2 required for %s filter" % btype)
            if "cutoff1" not in valid:
                raise TypeError("Butter:invalid cutoff1 argument")
            if "cutoff2" not in valid:
                raise TypeError("Butter:invalid cutoff2 argument")
            if cutoff1 > cutoff2:
                raise ValueError(
                    "Butter:cutoff1 must be less than or equal to cutoff2")
        else:
            raise ValueError("Butter: invalid btype %s" % btype)
        self.btype = btype
        # initialize base filter variables
        A = float(rolloff)
        fs = float(sampling)
        Oc = cutoff
        f1 = cutoff1
        f2 = cutoff2
        B = 99.99
        wp = .3 * np.pi
        ws = 2 * wp
        d1 = B / 100.0
        d2 = 10**(np.log10(d1) - (A / 20.0))
        self.N = int(np.ceil((np.log10(((1 / (d1**2)) - 1) /
                                       ((1 / (d2**2)) - 1))) / (2 * np.log10(wp / ws))))
        if self.N % 2 == 1:
            self.N += 1
        self.wc = 10**(np.log10(wp) - (1.0 / (2 * self.N))
                       * np.log10((1 / (d1**2)) - 1))
        self.fs = fs
        self.fc = Oc
        self.f1 = f1
        self.f2 = f2

        # to store the filtered data
        self.output = []
        # to store passed in data
        self.data = []
        # list of frequencies used in calculation of filters
        self.frequencylist = np.zeros((self.N // 2 + 1, 5))

        # set variables for desired filter
        self.filter = {
            "lowpass": self.__lowpass_filter_variables,
            "highpass": self.__highpass_filter_variables,
            "bandpass": self.__bandpass_filter_variables,
            "notch": self.__notch_filter_variables,
            "bandstop": self.__bandstop_filter_variables
        }[btype]()

    def filtfilt(self):
        """Returns accumulated output values with forward-backwards filtering
        @return list of float/int accumulated output values, filtered through forward-backward filtering
        """
        tempfrequencylist = [
            [0 for i in range(5)] for j in range(self.N // 2 + 1)]
        data = self.output[:]
        data.reverse()
        for i in range(len(data)):
            data[i] = __filterHelper(data[i], tempfrequencylist)
        data.reverse()
        return data

    def send(self, data):
        """Send data to Butterworth filter
        @param data list of floats amplitude data to take in
        @return values from the filtered data, with forward filtering
        """
        if type(data) != list:
            raise TypeError(
                "Butter.send: type of data must be a list of floats")
        self.data += data
        output = []
        for amplitude in data:
            newamp = _filterHelper(
                amplitude, self.frequencylist, self.filter, self.N)
            output.append(newamp)
        self.output += output
        return output

    def __basic_filter_variables(self):
        """Returns basic filter variables
        @return dictionary key:string variable value: lambda k
        """
        basic = np.zeros((9, (self.N // 2)))
        for k in range(self.N // 2):
            a = self.wc * \
                np.sin((float(2.0 * (k + 1) - 1) / (2.0 * self.N)) * np.pi)
            B = 1 + a + ((self.wc**2) / 4.0)
            basic[0][k] = (self.wc**2) / (4.0 * B)
            basic[1][k] = 2
            basic[2][k] = 1
            basic[3][k] = 0
            basic[4][k] = 0
            basic[5][k] = 2 * ((self.wc**2 / (4.0)) - 1) / (B)
            basic[6][k] = (
                1 - a + (self.wc**2 / (4.0))) / (B)
            basic[7][k] = 0
            basic[8][k] = 0

        return basic

    def __lowpass_filter_variables(self):
        """Returns lowpass filter variables
        @return dictionary key:string variable value: lambda k
        """
        basic = self.__basic_filter_variables()

        Op = 2 * (np.pi * self.fc / self.fs)
        vp = 2 * np.arctan(self.wc / 2.0)

        alpha = np.sin((vp - Op) / 2.0) / \
            np.sin((vp + Op) / 2.0)

        lowpass = np.zeros((9, (self.N // 2)))
        for k in range(self.N // 2):
            C = 1 - basic[5][k] * \
                alpha + basic[6][k] * (alpha**2)
            a = self.wc * \
                np.sin((float(2.0 * (k + 1) - 1) / (2.0 * self.N)) * np.pi)
            B = 1 + a + ((self.wc**2) / 4.0)
            lowpass[0][k] = ((1 - alpha)**2) * basic[0][k] / C
            lowpass[1][k] = basic[1][k]
            lowpass[2][k] = basic[2][k]
            lowpass[3][k] = basic[3][k]
            lowpass[4][k] = basic[4][k]
            lowpass[5][k] = (
                (1 + alpha**2) * basic[5][k] - 2 * alpha * (1 + basic[6][k])) / C
            lowpass[6][k] = (
                alpha**2 - basic[5][k] * alpha + basic[6][k]) / C
            lowpass[7][k] = basic[7][k]
            lowpass[8][k] = basic[8][k]
        return lowpass

    def __highpass_filter_variables(self):
        """Returns highpass filter variables
        @return dictionary key:string variable value: lambda k
        """
        basic = self.__basic_filter_variables()
        Op = 2 * (np.pi * float(self.fc) / self.fs)
        vp = 2 * np.arctan(self.wc / 2.0)

        alpha = -(np.cos((vp + Op) / (2.0))) / \
            (np.cos((vp - Op) / (2.0)))

        highpass = np.zeros((9, (self.N // 2)))
        for k in range(self.N // 2):
            C = 1 - basic[5][k] * \
                alpha + basic[6][k] * (alpha**2)
            highpass[0][k] = ((1 - alpha)**2) * basic[0][k] / C
            highpass[1][k] = -basic[1][k]
            highpass[2][k] = basic[2][k]
            highpass[3][k] = basic[3][k]
            highpass[4][k] = basic[4][k]
            highpass[5][k] = (
                -(1.0 + alpha**2) * basic[5][k] + 2 * alpha * (1 + basic[6][k])) / C
            highpass[6][k] = (
                float(alpha**2) - basic[5][k] * alpha + basic[6][k]) / C
            highpass[7][k] = basic[7][k]
            highpass[8][k] = basic[8][k]
        return highpass

    def __bandpass_filter_variables(self):
        """Returns bandpass filter variables
        @return dictionary key:string variable value: lambda k
        """
        basic = self.__basic_filter_variables()
        Op1 = 2 * (np.pi * (self.f1) / self.fs)
        Op2 = 2 * (np.pi * (self.f2) / self.fs)
        alpha = np.cos((Op2 + Op1) / 2.0) / np.cos((Op2 - Op1) / 2.0)
        k = (self.wc / 2.0) / np.tan((Op2 - Op1) / 2.0)
        A = 2 * alpha * k / (k + 1)
        B = (k - 1) / (k + 1)

        bandpass = np.zeros((9, (self.N // 2)))
        for k in range(self.N // 2):
            C = 1 - basic[5][k] * B + basic[6][k] * (B**2)

            bandpass[0][k] = basic[0][k] * ((1 - B)**2) / C
            bandpass[1][k] = 0
            bandpass[2][k] = -basic[1][k]
            bandpass[3][k] = 0
            bandpass[4][k] = basic[2][k]
            bandpass[5][k] = (A / C) * (B * (basic[5][k] -
                                             2 * basic[6][k]) + (basic[5][k] - 2))
            bandpass[6][k] = (1 / C) * ((A**2) * (1 - basic[5][k] + basic[6][k]) +
                                        2 * B * (1 + basic[6][k]) - basic[5][k] * (B**2) - basic[5][k])
            bandpass[7][k] = (A / C) * (B * (basic[5][k] - 2) +
                                        (basic[5][k] - 2 * basic[6][k]))
            bandpass[8][k] = (1 / C) * ((B**2) - basic[5][k] * B + basic[6][k])
        return bandpass

    def __notch_filter_variables(self):
        """Returns notch filter variables
        @return dictionary key:string variable value: lambda k
        """
        basic = self.__basic_filter_variables()
        x = 1.0
        f1 = (1.0 - (x / 100)) * self.fc
        f2 = (1.0 + (x / 100)) * self.fc
        Op1 = 2 * (np.pi * f1 / self.fs)
        Op2 = 2 * (np.pi * f2 / self.fs)
        alpha = np.cos((Op2 + Op1) / 2.0) / np.cos((Op2 - Op1) / 2.0)
        k = (self.wc / 2.0) * np.tan((Op2 - Op1) / 2.0)
        A = 2 * alpha / (k + 1)
        B = (1 - k) / (1 + k)

        notch = np.zeros((9, (self.N // 2)))
        for k in range(self.N // 2):
            C = 1 + basic[5][k] * \
                B + basic[6][k] * (B**2)
            notch[0][k] = basic[0][k] * ((1 + B)**2) / C
            notch[1][k] = -4.0 * A / (B + 1)
            notch[2][k] = 2.0 * ((2 * (A**2)) / ((B + 1)**2) + 1)
            notch[3][k] = -4.0 * A / (B + 1)
            notch[4][k] = 1
            notch[5][k] = -(A / C) * \
                (B * (basic[5][k] + 2 * basic[6][k]) +
                 (2 + basic[5][k]))
            notch[6][k] = (1 / C) * \
                ((A**2) * (1 + basic[5][k] + basic[6][k]) +
                 2 * B * (1 + basic[6][k]) +
                 basic[5][k] * (B**2) +
                 basic[5][k])
            notch[7][k] = -(A / C) * \
                (B * (basic[5][k] + 2) +
                 (basic[5][k] + 2 * basic[6][k]))
            notch[8][k] = (1 / C) * \
                ((B**2) +
                 basic[5][k] * B +
                 basic[6][k])
        return notch

    def __bandstop_filter_variables(self):
        """Returns bandstop filter variables
        @return dictionary key:string variable value: lambda k
        """
        basic = self.__basic_filter_variables()
        Op1 = 2 * (np.pi * self.f1 / self.fs)
        Op2 = 2 * (np.pi * self.f2 / self.fs)
        alpha = np.cos((Op2 + Op1) / 2.0) / np.cos((Op2 - Op1) / 2.0)
        k = (self.wc / 2.0) * np.tan((Op2 - Op1) / 2.0)
        A = 2 * alpha / (k + 1)
        B = (1 - k) / (1 + k)

        bandstop = np.zeros((9, self.N // 2))
        for k in range(self.N // 2):
            C = 1 + basic[5][k] * \
                B + basic[6][k] * (B**2)
            bandstop[0][k] = basic[0][k] * ((1 + B)**2) / C
            bandstop[1][k] = -4.0 * A / (B + 1)
            bandstop[2][k] = 2.0 * ((2 * (A**2)) / ((B + 1)**2) + 1)
            bandstop[3][k] = -4.0 * A / (B + 1)
            bandstop[4][k] = 1
            bandstop[5][k] = -(A / C) * \
                (B * (basic[5][k] + 2 * basic[6][k]) +
                 (2 + basic[5][k]))
            bandstop[6][k] = (1 / C) * \
                ((A**2) * (1 + basic[5][k] + basic[6][k]) +
                 2 * B * (1 + basic[6][k]) +
                 basic[5][k] * (B**2) +
                 basic[5][k])
            bandstop[7][k] = -(A / C) * \
                (B * (basic[5][k] + 2) +
                 (basic[5][k] + 2 * basic[6][k]))
            bandstop[8][k] = (1 / C) * \
                ((B**2) +
                 basic[5][k] * B +
                 basic[6][k])
        return bandstop
