import tensorflow as tf

# fft_size = n_fft
# frame step = hop-length


class Spectrogram(tf.keras.layers.Layer):
    """Converts batches of normalized audio tensors to spectrograms."""
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        all_params = {**params, **hparams}
        self.frame_size = all_params['frame_size']
        self.hop_length = all_params['hop_length']
        self.n_fft = all_params['n_fft']

    def call(self, audio_tensors):
        # Waveform input shape: (batch_size, num_samples) in the range [-1, 1]
        stfts = tf.signal.stft(
            audio_tensors,
            frame_length=self.frame_size,
            frame_step=self.hop_length,
            fft_length=self.n_fft
        )
        # The STFT returns a real (magnitude) and complex (phase) component
        return tf.abs(stfts)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frame_size': self.frame_size,
            'hop_length': self.hop_length,
            'n_fft': self.n_fft,
        })
        return config


class MelSpectrogram(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        all_params = {**params, **hparams}
        self.sample_rate = all_params['sample_rate']
        self.n_fft = all_params['n_fft']
        self.n_mels = all_params['n_mels']
        self.min_freq = all_params['min_freq']
        self.max_freq = all_params['max_freq']
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.min_freq,
            upper_edge_hertz=self.max_freq
        )

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(MelSpectrogram, self).build(input_shape)

    def call(self, spectrograms):
        # Warp the linear scale spectrograms into the mel-scale.
        return tf.tensordot(spectrograms, self.mel_filterbank, 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
            'sample_rate': self.sample_rate,
            'min_freq': self.min_freq,
            'max_freq': self.max_freq,
        })
        return config


class LogMelSpectrogram(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)

    def call(self, mel_spectrograms):
        return tf.math.log(mel_spectrograms + 1e-6)

    def get_config(self):
        return super().get_config().copy()