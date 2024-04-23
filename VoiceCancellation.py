import numpy as np
import librosa
import scipy
import soundfile as sf
from scipy import signal

from TTS.api import TTS, load_config


def mix_audio(audio_path, noise_path, mixed_path, noise_level=1):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)
    # Load the noise file
    noise, sr_noise = librosa.load(
        noise_path, sr=sr
    )  # Ensure noise is resampled to match audio

    # If noise is shorter than the audio, repeat the noise
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    # Trim noise to the length of audio
    noise = noise[: len(audio)]

    # Mix audio with noise
    mixed_signal = audio + noise_level * noise

    # Normalize mixed signal to prevent clipping
    mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))

    # Save the mixed audio
    sf.write(mixed_path, mixed_signal, sr)


def exec_mix(audio_path, noise_path, mixed_path):

    # Call the function
    mix_audio(
        audio_path, noise_path, mixed_path, noise_level=0.5
    )  # Adjust noise_level as needed


def exec_cancellation_2(mixed_path, noise_path, denoisy_path):

    # Load your files
    noisy_signal, sr = librosa.load(mixed_path, sr=None)
    noise, sr_noise = librosa.load(noise_path, sr=None)

    # Ensure both files have the same sample rate
    if sr != sr_noise:
        raise ValueError("Sample rates do not match!")

    w = noisy_signal
    s = librosa.stft(w)  # Short-time Fourier transform
    ss = np.abs(s)  # get magnitude
    angle = np.angle(s)  # get phase
    b = np.exp(1.0j * angle)  # use this phase information when Inverse Transform

    nw = noise
    ns = librosa.stft(nw)
    nss = np.abs(ns)
    mns = np.mean(nss, axis=1)  # get mean

    # subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
    sa = ss - mns.reshape((mns.shape[0], 1))  # reshape for broadcast to subtract
    sa0 = sa * b  # apply phase information
    y = librosa.istft(sa0)  # back to time domain signal

    # save as a wav file
    scipy.io.wavfile.write(
        denoisy_path, sr, (y * 32768).astype(np.int16)
    )  # save signed 16-bit WAV format
    # librosa.output.write_wav(outfile, y , sr)  # save 32-bit floating-point WAV format, due to y is float
    print("write wav", denoisy_path)


def exec_cancellation_3(mixed_path, noise_path, denoisy_path):
    # Load your files
    noisy_signal, sr = librosa.load(mixed_path, sr=None)
    noise, sr_noise = librosa.load(noise_path, sr=None)

    # Ensure both files have the same sample rate
    if sr != sr_noise:
        raise ValueError("Sample rates do not match!")

    # Length adjustment
    min_len = min(len(noisy_signal), len(noise))
    noisy_signal = noisy_signal[:min_len]
    noise = noise[:min_len]

    # Perform STFT
    s = librosa.stft(noisy_signal)
    ns = librosa.stft(noise)

    # Magnitude and phase
    ss = np.abs(s)
    angle = np.angle(s)
    nss = np.abs(ns)

    # Noise estimation
    mns = np.mean(nss, axis=1)

    # Spectral subtraction
    sa = np.maximum(ss - mns[:, np.newaxis], 0)  # Subtract and clip negative values

    # Reconstruct signal using original phase
    sa0 = sa * np.exp(1.0j * angle)
    y = librosa.istft(sa0)

    # Save the denoised signal
    scipy.io.wavfile.write(denoisy_path, sr, (y * 32768).astype(np.int16))
    print("write wav", denoisy_path)


def exec_cancellation(mixed_path, noise_path, denoisy_path):
    """Given that you already have both the noise file and the combined noise + audio file, the approach to noise cancellation can be more straightforward and controlled. In such a scenario, you can employ techniques like spectral subtraction, adaptive filtering, or even deep learning models trained specifically to separate the noise identified in your noise file from your audio. Below, I'll outline a step-by-step guide to address this using spectral subtraction, which is one of the simpler methods to implement, and also a more advanced approach using deep learning.

    Spectral Subtraction Method
    Spectral subtraction is a method that estimates the magnitude of the noise in the frequency domain and subtracts it from the noisy signal to approximate the clean signal. Here’s how you can implement it:

    Load the audio files: Read both your noise file and your noise + audio file.
    Perform STFT (Short Time Fourier Transform): Transform both signals from time domain to frequency domain.
    Estimate the noise spectrum: Average the spectral frames of your noise file to get a noise spectrum estimate.
    Subtract the noise spectrum from the noisy signal spectrum: This will give you an estimate of the clean speech spectrum.
    Perform Inverse STFT: Convert the cleaned frequency domain signal back to the time domain.

    Advanced Method: Deep Learning
    If you find that traditional methods like spectral subtraction aren't effective enough (often due to non-linearities and variance in noise), using a deep learning model can be more robust. This requires more setup, including potentially training a model if pre-trained models aren't available for your specific type of noise.

    Data Preparation: Prepare your data by aligning your noise and noisy audio files into training pairs. If you have multiple instances or variations, this can help improve model robustness.
    Model Selection: Choose a model architecture. RNNs, CNNs, and autoencoder structures are popular for this task.
    Training: Train the model using a regression framework where the input is the noisy audio and the target is the clean audio (obtained by subtracting your noise from the combined file if possible).
    Inference: Apply the model to new noisy audio samples to perform noise cancellation.
    Frameworks like TensorFlow, PyTorch, and toolkits like SpeechBrain provide functionalities to implement these models. Depending on your specific requirements and available data, you might need to experiment with different configurations and parameters.

    Each of these methods has its pros and cons, and the best choice depends on the specific characteristics of your audio files and the quality of noise cancellation you need.
    """

    # Load your files
    noisy_signal, sr = librosa.load(mixed_path, sr=None)
    noise, sr_noise = librosa.load(noise_path, sr=None)

    # Ensure both files have the same sample rate
    if sr != sr_noise:
        raise ValueError("Sample rates do not match!")

    # STFT of signals
    stft_noise = librosa.stft(noise)
    stft_noisy_signal = librosa.stft(noisy_signal)
    # stft_noise = signal.stft(noise)
    # stft_noisy_signal = signal.stft(noisy_signal)

    # Average noise spectrum
    avg_noise_spectrum = np.mean(np.abs(stft_noise), axis=1, keepdims=True)

    # Subtract noise spectrum
    subtracted_spectrum = np.abs(stft_noisy_signal) - avg_noise_spectrum
    subtracted_spectrum = np.maximum(subtracted_spectrum, 0)  # Remove negative values

    # Get phase of noisy signal
    phase = np.angle(stft_noisy_signal)

    # Reconstruct signal
    reconstructed_signal = librosa.istft(subtracted_spectrum * np.exp(1j * phase))

    # Save the cleaned signal
    sf.write(denoisy_path, reconstructed_signal, sr)


def apply_wiener_filter(mixed_path, noise_path, denoisy_path):
    noisy_signal, sr = librosa.load(mixed_path, sr=None)
    noise_signal, _ = librosa.load(noise_path, sr=sr)  # Ensure same sample rate
    # STFT
    stft_signal = librosa.stft(noisy_signal)
    stft_noise = librosa.stft(noise_signal)

    # Calculate magnitude and phase
    mag_signal = np.abs(stft_signal)
    phase_signal = np.angle(stft_signal)

    # Estimate noise power spectrum
    mag_noise = np.abs(stft_noise)
    power_noise = np.mean(
        mag_noise**2, axis=1, keepdims=True
    )  # Average noise power spectrum

    # Estimate signal power spectrum
    power_signal = mag_signal**2

    # Calculate SNR
    snr = power_signal / (power_noise + 1e-10)  # Add epsilon to avoid division by zero

    # Wiener gain
    wiener_gain = snr / (snr + 1)
    filtered_magnitude = mag_signal * wiener_gain

    # ISTFT to reconstruct the signal
    filtered_stft = filtered_magnitude * np.exp(1j * phase_signal)
    recovered_signal = librosa.istft(filtered_stft)
    # Save the cleaned signal
    sf.write(denoisy_path, recovered_signal, sr)
    # librosa.output.write_wav(denoisy_path, recovered_signal, sr)


import librosa
import numpy as np
import soundfile as sf


def subtract_audio(mixed_path, single_speaker_path, output_path):
    # Load both audio files
    mixed_audio, sr = librosa.load(mixed_path, sr=None)
    single_audio, sr_single = librosa.load(single_speaker_path, sr=None)

    # Ensure the same sample rate for both files
    if sr != sr_single:
        raise ValueError("Sample rates do not match!")

    # Truncate the longer audio if they are not the same length
    min_length = min(len(mixed_audio), len(single_audio))
    mixed_audio = mixed_audio[:min_length]
    single_audio = single_audio[:min_length]

    # Subtract the single speaker's audio from the mixed audio
    result_audio = mixed_audio - single_audio

    # Write the resulting audio back to a file
    sf.write(output_path, result_audio, sr)


def clean_audio(noisy_path, output_path):
    # Load both audio files
    noisy_signal, sr = librosa.load(noisy_path, sr=None)
    stft_noisy_signal = librosa.stft(noisy_signal)
    noisy_spectrum = np.abs(stft_noisy_signal)
    noise_spectrum = np.mean(stft_noisy_signal, axis=1, keepdims=True)
    alpha = 2
    clean_spectrum = np.maximum(noisy_spectrum - alpha * noise_spectrum, 0)
    clean_signal = librosa.istft(clean_spectrum)
    sf.write(output_path, clean_signal, sr)


def tts_saves_wav(output_path):
    model = "tts_models/en/ljspeech/vits--neon"
    tts = TTS(model, gpu=True).to("cuda")
    prompt = "Hello, how are you today ? I am glad to see you so willing to learn mathematics !"
    generated_audio = tts.tts(text=prompt)
    # # waveform = waveforms.squeeze(1).detach().numpy()[0]
    # generated_audio = np.array(generated_audio)
    # # Convert float32 data [-1,1] to int16 data [-32767,32767]
    # generated_audio = (generated_audio * 32767).astype(np.int16).tobytes()

    # Write the resulting audio back to a file
    sf.write(output_path, generated_audio, 22050)


if __name__ == "__main__":

    # Paths to your audio and noise files
    audio_path = "audios/test/Recording (13)_cropped.wav"
    noise_path = "audios/test/Recording (12)_cropped.wav"
    mixed_path = "audios/test/mixed_audio_3.wav"
    denoisy_path = "audios/test/denoisy_audio_6.wav"
    tts_output = "audios/test/tts_output.wav"
    tts_record = "audios/test/tts_output_SR22k.wav"

    # exec_mix(audio_path, noise_path, mixed_path)
    # exec_cancellation(mixed_path, noise_path, denoisy_path)
    # exec_cancellation_2(mixed_path, noise_path, denoisy_path)
    # exec_cancellation_3(mixed_path, noise_path, denoisy_path)

    # subtract_audio(tts_output, tts_record, denoisy_path)

    clean_audio(tts_record, "audios/test/clean_tts_output_SR22k.wav")

    # Usage
    # apply_wiener_filter(mixed_path, noise_path, denoisy_path)

    # tts_saves_wav(tts_output)
