import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile


class AudioFile:

    def __init__(self, wav_path):
        self.wav_path = wav_path
        self.data, self.sample_rate = librosa.load(self.wav_path)
        self.shape = self.data.shape

    def shift_pitch(self, factor=-5.0):
        """Permissible factor range - [-5.0, 5.0)"""
        self.data = librosa.effects.pitch_shift(self.data,
                                                self.sample_rate,
                                                n_steps=factor)

    def get_spectrogram(self, n_fft, hop_length, plot=False):
        fft_windows = librosa.stft(self.data,
                                   n_fft=n_fft,
                                   hop_length=hop_length)
        spectrogram = np.abs(fft_windows) ** 2

        if plot:
            plt.title('Spectrogram')
            librosa.display.specshow(spectrogram,
                                     sr=self.sample_rate,
                                     hop_length=hop_length,
                                     x_axis='time',
                                     y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.show()

        return spectrogram

    def get_melspectrogram(self, n_fft, hop_length, n_mels, plot=False):
        mel = librosa.feature.melspectrogram(self.data,
                                             sr=self.sample_rate,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mels=n_mels)
        logmel = librosa.power_to_db(mel, ref=np.max)

        if plot:
            plt.title('Log-Mel Spectrogram')
            librosa.display.specshow(logmel,
                                     sr=self.sample_rate,
                                     hop_length=hop_length,
                                     x_axis='time',
                                     y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.show()

        return logmel

    def save_wav(self, path):
        soundfile.write(path, self.data, self.sample_rate)


