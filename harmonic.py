import copy
import numpy as np
import pandas as pd
from scipy.optimize import  basinhopping
from easier import ParamState
from sklearn.linear_model import LinearRegression


class Harmonic:
    def __init__(self, freq=1, num_freqs=1, refine_fundamental=True):
        """
        A class for fitting and manipulating harmonic series
        :param freq:  The fundamental frequency of the series.  If do_freq_fit is positive,
                      this is used as a guess to identify the actual fundamental.
        :param num_freqs:  The number of frequencies to include in the series
        :param do_freq_fit:  If set to True, will try to refine the fundamental estimate
        """
        self.num_freqs = num_freqs
        self.init_freq = freq
        self._refine_fundamental = refine_fundamental
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

    def derivative(self):
        """
        :return: A new Harmonic object that is the derivative of this one
        """
        h = copy.deepcopy(self)
        h.cosines = self.w_array * self.sines
        h.sines = - self.w_array * self.cosines
        return h

    def integral(self):
        """
        :return: A new Harmonic object that is the integral of this one
        """
        h = copy.deepcopy(self)
        h.cosines = - self.sines / self.w_array
        h.sines = self.cosines / self.w_array
        return h

    def refine_frequency(self, time, amplitude):
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
        xf = basinhopping(cost, x0, minimizer_kwargs=dict(args=(p,))).x
        p.ingest(xf)
        if (p.f - self.f0) > 3 ** 2:
            raise ValueError(f'Guess freq: {self.f0}, Fit Freq: {p.f}  too far apart')
        self.f0 = p.f

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

    def _fit_params(self, times, values):
        """
        Regresses a timeseries against the basis functions
        :param times:
        :param values:
        :return:
        """
        basis = self._get_bases(times)

        # in case you want to do ridge regression
        # alpha = .000002
        # model_res = Ridge(alpha=alpha)

        # in case you want to do just linear regression
        model = LinearRegression()

        model.fit(basis, values)
        self.sines = model.coef_[:self.num_freqs]
        self.cosines = model.coef_[self.num_freqs:]
        return model.coef_

    def fit(self, times, values):
        """
        Fit a time series with the harmonic series
        :param times: An array of timestamps
        :param values:  An array of signal amplitudes
        :return:
        """
        values = values - np.mean(values)
        if self._refine_fundamental:
            self.refine_frequency(times, values)
        self._fit_params(times, values)

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
        return np.squeeze(prediction)