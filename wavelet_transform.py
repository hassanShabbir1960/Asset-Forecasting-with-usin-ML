import pywt

class WaveletTransform:

    def __init__(self, dataset_df, price_col, wavelet='db2', level=3):
        self.dataset_df = dataset_df
        self.price_col = price_col
        self.wavelet = wavelet
        self.level = level

    def apply_wavelet_transform(self):
        out = pywt.wavedecn(self.dataset_df[self.price_col], wavelet=self.wavelet, level=self.level)
        out, slices = pywt.coeffs_to_array(out, padding=0, axes=None)
        self.dataset_df['wt'] = out[:1989]
        return self.dataset_df
