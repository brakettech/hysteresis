import numpy as np
import pandas as pd
from daq.pico import CSV
from scipy.optimize import fmin, minimize, basinhopping, fsolve
from easier import Item
from harmonic import Harmonic


class HarmonicAnalyzer:
    def __init__(self, params):
        self.params = params
        self.df = None
        self.df_fit = None
        self.file_name = None
        self.delta = None
        self.channel_mapper = dict(
            a='sig_gen',
            b='res_volt',
            c='sec_volt',
        )

    def set_channel_names(self, **kwargs):
        self.channel_mapper.update(kwargs)
        return self

    def normalizer(self, y, scale=None, return_scale=False):
        if scale is None:
            scale = 1. / (np.sqrt(2) * np.std(y))
        if return_scale:
            return scale * y, scale
        else:
            return scale * y

    def demeaner(self, df):
        for col in df.columns:
            if col == 't':
                continue
            df.loc[:, col] = df.loc[:, col] - df.loc[:, col].mean()
        return df

    def fit(self, file_name, fundamental_freq):
        self.file_name = file_name
        self.df, self.df_fit = self._get_fit_frames(file_name, fundamental_freq)

    def _get_phase_delta(self, h_obj1, h_obj2, t_ref):
        zero1 = fsolve(h_obj1.predict, t_ref)
        zero2 = fsolve(h_obj2.predict, zero1)
        delta = zero2 - zero1
        return delta

    def load(self, file_name):
        # load the file and make sure all data has zero-mean
        kwargs = dict(
            max_sample_freq=self.params.max_sample_freq
        )
        kwargs.update(self.channel_mapper)
        df = CSV(file_name, **kwargs).df
        df = self.demeaner(df)
        self.df_data = df
        return df

    def _get_fit_frames(self, file_name, fundamental_freq):
        # make a local reference to the params
        params = self.params

        df = self.load(file_name)

        # refine the guess of the fundamental frequency
        h = Harmonic(freq=fundamental_freq, num_freqs=1)
        f0 = h.refine_frequency(df.t, df.sig_gen, simple=params.simple_freq_fit).f0

        # initialize harmonic objects for resistor and secondary voltages
        h_resistor = Harmonic(freq=f0, num_freqs=params.num_freqs)
        h_secondary = Harmonic(freq=f0, num_freqs=params.num_freqs)

        # fit the harmonic objects to the data
        h_resistor.fit(df.t, df.res_volt, method=params.method, alpha=params.alpha)
        h_secondary.fit(df.t, df.sec_volt, method=params.method, alpha=params.alpha)

        # create an harmonic object for the actual/expected secondary
        h_expected = -h_resistor.derivative()
        h_actual = h_secondary.clone()

        # set a reference time to be near the center of the dataset centered on
        # an ascending zero crossing of the expected voltage
        t_ref = df.t.iloc[len(df) // 2]
        t_ref = fsolve(h_expected.predict, t_ref)[0]
        if h_expected.derivative().predict(t_ref) < 0:
            t_ref = t_ref + 0.5 * h.period
        self.t_ref = t_ref

        # take derivatives so that I can set the phases so that actual/expected have peaks at same time
        h_actual_deriv = h_actual.derivative()
        h_expected_deriv = h_expected.derivative()

        # normalize all signals to have fits of amplitudes of about 1
        res_scale = sec_scale = 1
        if params.normalize:
            # run a fit for the entire input dataframe
            res_fit = h_resistor.predict(df.t)
            sec_fit = h_secondary.predict(df.t)

            # find the scale parameters from the fits
            _, res_scale = self.normalizer(res_fit, return_scale=True)
            _, sec_scale = self.normalizer(sec_fit, return_scale=True)

            # normalize the raw data by the fit scales
            df.loc[:, 'res_volt'] = res_scale * df.res_volt
            df.loc[:, 'sec_volt'] = sec_scale * df.sec_volt

        self.index_mu = (res_scale / sec_scale) / f0

        # limit the dataframe to be only the number of periods anchored to t_ref
        df = df[(df.t >= t_ref) & (df.t < t_ref + params.periods * h.period)].reset_index(drop=True)

        # compute a delta that will make it so that the peaks of actual and expected signals
        # coincide as closely as possible
        delta1 = self._get_phase_delta(h_expected_deriv, h_actual_deriv, t_ref + .2 * h.period)
        delta2 = self._get_phase_delta(h_expected_deriv, h_actual_deriv, t_ref + .7 * h.period)
        self.delta = delta = .5 * (delta1 + delta2)
        self.index_rho = h.f0 * self.delta[0]

        # make fits for actual signal and phase adjusted expected signal
        actual_secondary = h_secondary.predict(df.t)
        expected_secondary = h_expected.predict(df.t - delta)

        actual_scale = expected_scale = 1.
        # normalize the fits if requested
        if params.normalize:
            actual_secondary, actual_scale = self.normalizer(actual_secondary, return_scale=True)
            expected_secondary, expected_scale = self.normalizer(expected_secondary, return_scale=True)

        # create a dataframe of the fits
        kwargs = dict(
            t=df.t,
            actual=actual_secondary,
            expected=expected_secondary,
        )
        df_fit = pd.DataFrame(kwargs, columns=kwargs.keys())

        t0_expected_1 = fsolve(lambda t: h_expected.predict(t - delta), self.t_ref)[0]
        t0_expected_2 = fsolve(lambda t: h_expected.predict(t - delta), self.t_ref + 0.5 * h.period)[0]

        y0_actual_1 = h_actual.predict(t0_expected_1)
        y0_actual_2 = h_actual.predict(t0_expected_2)
        y0_expected_1 = h_expected.predict(t0_expected_1 - delta)
        y0_expected_2 = h_expected.predict(t0_expected_2 - delta)

        self.ref_points = Item(
            t0_expected_1=t0_expected_1,
            t0_expected_2=t0_expected_2,
            y0_actual_1=y0_actual_1 * actual_scale,
            y0_actual_2=y0_actual_2 * actual_scale,
            y0_expected_1=y0_expected_1 * expected_scale,
            y0_expected_2=y0_expected_2 * expected_scale,
        )
        self.index_hyst = abs(y0_actual_2 * actual_scale) + abs(y0_actual_1 * actual_scale)

        # return the frames with raw data and fits
        return df, df_fit


