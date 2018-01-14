import copy
import numpy as np
import pandas as pd
from scipy.optimize import fmin_powell
from easier import ParamState
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from scipy.fftpack import fft


class Harmonic:
    def __init__(self, harmonics=None):
        if harmonics is None:
            harmonics = [1]
        self.harmonics = np.array(harmonics)
        num_freqs = len(harmonics)
        self._sines = np.ones((num_freqs, 1))
        self._cosines = np.ones((num_freqs, 1))
        self._z = np.ones((num_freqs, 1))
        self.f0 = 1
        self.intercept = 0

    @property
    def sines(self):
        return self._sines

    @sines.setter
    def sines(self, new):
        self._sines = new
        self._z  = self.cosines + 1j * self.sines

    @property
    def cosines(self):
        return self._cosines

    @cosines.setter
    def cosines(self, new):
        self._cosines = new
        self._z  = self.cosines + 1j * self.sines

    @property
    def z(self):
        return self._z
        # return self.cosines + 1j * self.sines

    @z.setter
    def z(self, new):
        self._z = new
        self._cosines = np.real(new)
        self._sines = np.imag(new)

    @property
    def amplitudes(self):
        return np.sqrt(self.sines ** 2 + self.cosines ** 2)

    @property
    def phases(self):
        return np.arctan2(self.sines, self.cosines)


    @property
    def num_freqs(self):
        return len(self.harmonics)

    @property
    def f0(self):
        """
        :return: The fundamental frequency
        """
        return self._f0

    @f0.setter
    def f0(self, f):
        """
        Set the fundamental frequency
        :param f: The frequency to set
        """
        self._f0 = f
        self._w0 = 2 * np.pi * f

    @property
    def w0(self):
        """
        :return: The fundamental frequency in angular units
        """
        return self._w0

    @w0.setter
    def w0(self, w):
        """
        Set the angular frequency
        :param w:  the frequency to set
        :return:
        """
        self._w0 = w
        self._f0 = w / (2 * np.pi)

    @property
    def period(self):
        return 1. / self.f0

    @period.setter
    def period(self, period_val):
        self.f0 = 1. / period_val

    @property
    def w(self):
        """
        :return: An array of angular frequencies used.
        """
        return np.array([n * self.w0 for n in self.harmonics])

    @property
    def f(self):
        """
        :return: An array of frequencies used
        """
        return np.array([n * self.f0 for n in self.harmonics])

    def _get_padded_length(self, initial_length, interp_exp=0):
        for nn in range(int(1e6)):
            padded_length = 2 ** nn
            if padded_length >= initial_length:
                break
        return padded_length * 2 ** interp_exp

    def get_freq(self, time, amplitude, interp_exp=3):
        # demean the signal
        amplitude = amplitude - np.mean(amplitude)

        # pad length to power of two with maybe some interpolation
        padded_length = self._get_padded_length(len(amplitude), interp_exp=interp_exp)

        # get the sample time
        dt = np.median(np.diff(time))

        # compute the fft
        z = fft(amplitude, n=padded_length)

        # define a slice for postive frequencies
        ind = slice(0, int((len(z) / 2)))

        # get positive amplitudes
        amp_f = np.abs(z)[ind]

        # compute positive freqs
        f = np.fft.fftfreq(len(z), d=dt)[ind]

        # return the max freq
        return f[np.where(amp_f == np.max(amp_f))[0]][0]

    def refine_frequency(self, time, amplitude, guess, verbose=False):
        # set up to do do a mininzer fit to best freq
        p = ParamState(
            't',
            'y_true',
            a=1,
            b=1,
            f=guess,
        )
        p.given(
            t=time,
            y_true=amplitude
        )

        def model(p):
            return (
                p.a * np.sin(2 * np.pi * p.f * p.t) +
                p.b * np.cos(2 * np.pi * p.f * p.t)
            )

        def cost(args, p):
            p.ingest(args)
            err = model(p) - p.y_true
            energy = np.sum(err ** 2)
            return energy

        x0 = p.array
        xf = fmin_powell(cost, x0, args=(p,), disp=verbose)
        p.ingest(xf)
        if (p.f - guess) > 3 ** 2:
            raise ValueError(f'Guess freq: {self.f0}, Fit Freq: {p.f}  too far apart')
        return p.f

    def _ensure_same(self, other):
        if len(self.sines) != len(other.sines):
            raise ValueError('You can only place operators between objects of same dimensions')

        if self.f0 != other.f0:
            raise ValueError('You can only place operators between objects with same fundamental freq')

    def __add__(self, other):
        self._ensure_same(other)
        h = self.clone()
        h.cosines = self.cosines + other.cosines
        h.sines = self.sines + other.sines
        return h

    def __sub__(self, other):
        self._ensure_same(other)
        h = self.clone()
        h.cosines = self.cosines - other.cosines
        h.sines = self.sines - other.sines
        return h

    def __neg__(self):
        h = self.clone()
        h.cosines = -self.cosines
        h.sines = -self.sines
        return h

    def __div__(self, other):
        h = self.clone()
        h.z = self.z / other.z
        return h

    def __truediv__(self, other):
        return self.__div__(other)

    def __mul__(self, other):
        h = self.clone()
        h.z = self.z * other.z
        return h

    def clone(self):
        return copy.deepcopy(self)

    def derivative(self, order=1):
        h = self.clone()
        for nn in range(order):
            h = h._derivative()
        return h

    def _derivative(self):
        """
        :return: A new Harmonic object that is the derivative of this one
        """
        h = self.clone()
        h.cosines = self.w * self.sines
        h.sines = - self.w * self.cosines
        return h

    def integral(self, order=1):
        h = self.clone()
        for nn in range(order):
            h = h._integral()
        return h

    def _integral(self):
        """
        :return: A new Harmonic object that is the integral of this one
        """
        h = copy.deepcopy(self)
        h.cosines = - self.sines / self.w
        h.sines = self.cosines / self.w
        return h

    def _get_bases(self, time):
        """
        Creates basis functions for fitting
        """
        if not isinstance(time, np.ndarray):
            time = np.array(time)
        df_cos = pd.DataFrame(index=range(len(time)))
        df_sin = pd.DataFrame(index=range(len(time)))

        for n, w in enumerate(self.w):
            df_sin.loc[:, n] = np.sin(w * time)
            df_cos.loc[:, n] = np.cos(w * time)

        sin_bases, cos_bases = df_sin.values, df_cos.values
        return np.append(sin_bases, cos_bases, axis=1)

    def _fit_params(self, times, values, alpha=None, method='regression'):
        """
        Regresses a timeseries against the basis functions
        :param times: times to fit
        :param values:  values to fit
        :param alpha:  alpha for ridge or lasso regressors
        :param method: the regression method to use
        :return:  An array of coefficients
        """
        guess = self.get_freq(times, values, interp_exp=3)
        self.f0 = self.refine_frequency(times, values, guess, verbose=False)

        method_dict = {
            'regression': (LinearRegression, dict()),
            'ridge': (Ridge, dict(alpha=alpha)),
            'lasso': (Lasso, dict(alpha=alpha)),
            'lassocv': (LassoCV, dict())
        }
        if method not in method_dict.keys():
            raise ValueError(f'Method must be one of {list(method_dict.keys())}')

        basis = self._get_bases(times)

        model_class, kwargs = method_dict[method]
        model = model_class(**kwargs)

        model.fit(basis, values)
        self.sines = model.coef_[:self.num_freqs]
        self.cosines = model.coef_[self.num_freqs:]
        return model.coef_

    def fit(self, times, values, alpha=None, method='regression'):
        """
        Fit a time series with the harmonic series
        :param times: An array of timestamps
        :param values:  An array of signal amplitudes
        :return:
        """
        self.intercept = np.mean(values)
        values = values - self.intercept
        self._fit_params(times, values, alpha=alpha, method=method)
        return self

    def predict(self, t):
        """
        Use the harmonic information to predict (generate fit values) for timestamps
        :param t:
        :return:
        """
        # use mat mult to get args to trig funcs
        phi = np.matrix(t).T * np.matrix(self.w)

        # use mat mult to multiply by coeffs
        cosine_terms = np.cos(phi) * np.matrix(self.cosines).T
        sine_terms = np.sin(phi) * np.matrix(self.sines).T

        # cast to arrays
        cosine_terms = np.asarray(cosine_terms)
        sine_terms = np.asarray(sine_terms)

        # sum terms to get prediction
        prediction = np.sum(cosine_terms, axis=1)
        prediction += np.sum(sine_terms, axis=1)

        # return squeezed prediction
        out = np.squeeze(prediction) + self.intercept
        if not out.shape:
            out = float(out)
        return out
