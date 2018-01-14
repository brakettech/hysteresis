import copy
import numpy as np
import pandas as pd
from scipy.optimize import  basinhopping, fmin, fmin_powell
from easier import ParamState
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from pandashells.lib.lomb_scargle_lib import lomb_scargle


class Harmonic:
    def __init__(self, freq=1, num_freqs=1):
        """
        A class for fitting and manipulating harmonic series
        :param freq:  The fundamental frequency of the series.  If do_freq_fit is positive,
                      this is used as a guess to identify the actual fundamental.
        :param num_freqs:  The number of frequencies to include in the series
        :param do_freq_fit:  If set to True, will try to refine the fundamental estimate
        """
        self.num_freqs = num_freqs
        self.init_freq = freq
        self.sines = np.ones((num_freqs, 1))
        self.cosines = np.ones((num_freqs, 1))
        self.f0 = freq


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
    def w_array(self):
        """
        :return: An array of angular frequencies used.
        """
        return np.array([n * self.w0 for n in range(1, self.num_freqs + 1)])

    @property
    def f_array(self):
        """
        :return: An array of frequencies used
        """
        return np.array([n * self.f0 for n in range(1, self.num_freqs + 1)])

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
        h.cosines = self.w_array * self.sines
        h.sines = - self.w_array * self.cosines
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
        h.cosines = - self.sines / self.w_array
        h.sines = self.cosines / self.w_array
        return h

    def refine_frequency(self, time, amplitude, simple=False, verbose=False):
        # use lomb-scargle to get initial guess
        dfd = pd.DataFrame(dict(t=time, amp=amplitude))
        dfs = lomb_scargle(dfd, 't', 'amp', interp_exponent=1)
        freq = dfs[dfs.power == dfs.power.max()].freq

        # set up to do do a mininzer fit to best freq
        p = ParamState(
            't',
            'y_true',
            a=1,
            b=1,
            f=self.f0,
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
        if (p.f - self.f0) > 3 ** 2:
            raise ValueError(f'Guess freq: {self.f0}, Fit Freq: {p.f}  too far apart')
        self.f0 = p.f

        return self





    def refine_frequency_orig(self, time, amplitude, simple=False, verbose=False):
        """
        A method for refining the fundamental frequency of this harmonic
        :param time:  An array of timestamps
        :param amplitude:  An array of amplitudes ideally from a single tone signal
        :return:
        """
        p = ParamState(
            't',
            'y_true',
            a=1,
            f=self.f0,
            phi=0
        )
        p.given(
            t=time,
            y_true=amplitude
        )

        def model(p):
            return p.a * np.sin(2 * np.pi * p.f * p.t + p.phi)

        def cost(args, p):
            p.ingest(args)
            err = model(p) - p.y_true
            energy = np.sum(err ** 2)
            return energy

        x0 = p.array
        if simple:
            xf = fmin(cost, x0, args=(p,), disp=verbose)
        else:
            xf = basinhopping(cost, x0, minimizer_kwargs=dict(args=(p,))).x
        p.ingest(xf)
        if (p.f - self.f0) > 3 ** 2:
            raise ValueError(f'Guess freq: {self.f0}, Fit Freq: {p.f}  too far apart')
        self.f0 = p.f

        return self

    def _get_bases(self, time):
        """
        Creates basis functions for fitting
        """
        if not isinstance(time, np.ndarray):
            time = np.array(time)
        df_cos = pd.DataFrame(index=range(len(time)))
        df_sin = pd.DataFrame(index=range(len(time)))

        for n, w in enumerate(self.w_array):
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
        values = values - np.mean(values)
        self._fit_params(times, values, alpha=alpha, method=method)
        return self

    def predict(self, t):
        """
        Use the harmonic information to predict (generate fit values) for timestamps
        :param t:
        :return:
        """
        # use mat mult to get args to trig funcs
        phi = np.matrix(t).T * np.matrix(self.w_array)

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
        out = np.squeeze(prediction)
        if not out.shape:
            out = float(out)
        return out
