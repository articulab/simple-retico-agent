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
    # print(len(noise))
    # print(len(audio))
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    # Trim noise to the length of audio
    noise = noise[: len(audio)]

    # Mix audio with noise
    mixed_signal = audio + noise_level * noise

    # Normalize mixed signal to prevent clipping
    # print("normalize factor = ", np.max(np.abs(mixed_signal)))
    # mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))

    # print(noise[100000:100080])
    # print(audio[100000:100080])
    # print(mixed_signal[100000:100080])
    # print(mixed_signal[100000:100080])

    # Save the mixed audio
    sf.write(mixed_path, mixed_signal, sr)


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


def wiener_filter_2(noisy_path, clean_path, output_path):
    noisy_signal, sr = librosa.load(noisy_path, sr=None)
    clean_signal, _ = librosa.load(clean_path, sr=None)
    noisy_stft = np.abs(librosa.stft(noisy_signal))
    clean_stft = np.abs(librosa.stft(clean_signal))
    wiener_filter = clean_stft**2 / (clean_stft**2 + noisy_stft**2)
    enhanced_stft = wiener_filter * noisy_stft
    enhanced_audio = librosa.istft(enhanced_stft)
    sf.write(output_path, enhanced_audio, sr)


def subtract_audio(mixed_path, single_speaker_path, output_path):
    # Load both audio files
    mixed_audio, sr = librosa.load(mixed_path, sr=None)
    single_audio, sr_single = librosa.load(single_speaker_path, sr=None)

    # Ensure the same sample rate for both files
    if sr != sr_single:
        raise ValueError("Sample rates do not match!")

    # Truncate the longer audio if they are not the same length
    min_length = min(len(mixed_audio), len(single_audio))
    print(len(mixed_audio))
    print(len(single_audio))
    mixed_audio = mixed_audio[:min_length]
    single_audio = single_audio[:min_length]

    noise_level = 1.0
    # noise_level = 1.15
    # noise_level = sum(np.abs(mixed_audio)) / sum(np.abs(single_audio))
    # print(noise_level)
    # Subtract the single speaker's audio from the mixed audio
    result_audio = mixed_audio - noise_level * single_audio

    print(sum(np.abs(mixed_audio)))
    print(sum(np.abs(single_audio)))
    print(sum(np.abs(single_audio * noise_level)))
    print(sum(np.abs(result_audio)))

    # print(sum(mixed_audio))
    # print(sum(single_audio))
    # print(sum(result_audio))
    # print(np.max(mixed_audio))
    # print(np.max(single_audio))
    # print(np.max(result_audio))
    # print(np.argmax(mixed_audio))
    # print(np.argmax(single_audio))
    # print(np.argmax(result_audio))
    # print(np.min(mixed_audio))
    # print(np.min(single_audio))
    # print(np.min(result_audio))
    # print(np.argmin(mixed_audio))
    # print(np.argmin(single_audio))
    # print(np.argmin(result_audio))
    # print(mixed_audio[100000:100080])
    # print(single_audio[100000:100080])
    # print(result_audio[100000:100080])

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

    # generated_audio = np.array(generated_audio)
    # # Convert float32 data [-1,1] to int16 data [-32767,32767]
    # generated_audio = (generated_audio * 32767).astype(np.int16).tobytes()

    # Write the resulting audio back to a file
    sf.write(output_path, generated_audio, 22050)


# import audio_sync
# def calculate_latency(ref_wav_path, act_wav_path):
#     LATENCY_THRESHOLD = 10
#     # This assumes the test audio played by the devices under
#     # test is audio_sync.DEFAULT_TEST_AUDIO, whose properties
#     # are given in audio_sync.DEFAULT_TEST_AUDIO_PROPERTIES.
#     latencies, dropouts = audio_sync.AnalyzeAudios(ref_wav_path, act_wav_path)

#     # Verify there are no dropouts and the latency is below the threshold.
#     assert [] == [x for x in latencies if x[1] >= LATENCY_THRESHOLD]
#     assert [] == dropouts


# import syncaudio
# from syncaudio import synchronize
# from syncaudio import read_audio
# def calculate_latency_2(wav_a, wav_b):
#     # self = read_audio(wav_a)
#     # other = read_audio(wav_b)
#     audio_a, _ = librosa.load(wav_a, sr=None)
#     audio_b, _ = librosa.load(wav_b, sr=None)

#     delay = synchronize(
#         audio_a,
#         audio_b,
#         window_size=1024,
#         overlap=0,
#         spectral_band=512,
#         temporal_band=43,
#         peaks_per_bin=7,
#     )

#     print(delay, "seconds")


from syncstart import file_offset


def calculate_latency_3(wav_a, wav_b):
    # self = read_audio(wav_a)
    # other = read_audio(wav_b)
    audio_a, _ = librosa.load(wav_a, sr=None)
    audio_b, _ = librosa.load(wav_b, sr=None)

    delay = file_offset(in1=audio_a, in2=audio_b)

    print(delay, "seconds")


if __name__ == "__main__":
    folder_path = "audios/test/"

    # Paths to your audio and noise files
    audio_path = "audios/test/Recording (13)_cropped.wav"
    noise_path = "audios/test/Recording (12)_cropped.wav"
    mixed_path = "audios/test/mixed_audio_2_synch.wav"
    denoisy_path = "audios/test/test_substraction_tts_pilot_pilot.wav"

    tts_output = "audios/test/tts_output_48k.wav"
    tts_output_playback = "audios/test/tts_output_playback_48k_sync.wav"
    tts_record = "audios/test/tts_output_SR22k.wav"

    mixed_audio_synch = "audios/test/mixed_audio_2_mono_sync.wav"
    mixed_pilot = "audios/test/mixed_audio_pilot.wav"

    rec_user = folder_path + "rec_marius_cropped.wav"
    rec_user_mono = folder_path + "rec_marius_cropped_mono.wav"

    # pilot denoisy mix 1
    mix_1 = folder_path + "mix_rec_marius_cropped_rec_agent_cropped.wav"
    mix_1_2 = folder_path + "mix_rec_marius_cropped_rec_agent_cropped.wav"
    rec_agent = folder_path + "rec_agent_cropped.wav"
    denoisy_mix_1 = folder_path + "denoisy_mix_1.wav"

    # pilot denoisy mix 2
    mix_2 = folder_path + "mix_rec_marius_cropped_rec_tts_output_48k_sync.wav"
    rec_tts = folder_path + "rec_tts_output_48k_sync.wav"
    denoisy_mix_2 = folder_path + "denoisy_mix_2.wav"

    # test denoisy rec
    rec = folder_path + "rec_overlap_withttsoutput_mono_sync.wav"
    rec_tts = folder_path + "rec_tts_output_48k_sync.wav"
    denoisy_test_1 = folder_path + "denoisy_test_1.wav"

    # mix_audio(
    #     rec_user,
    #     rec_agent,
    #     mix_1_2,
    # )
    # mix_audio(rec_user_mono, rec_tts, mix_2)
    # subtract_audio(
    #     mix_1_2,
    #     rec_agent,
    #     denoisy_mix_1,
    # )
    # subtract_audio(mix_2, rec_tts, denoisy_mix_2)
    subtract_audio(rec, rec_tts, denoisy_test_1)

    # exec_cancellation(mixed_path, noise_path, denoisy_path)
    # exec_cancellation_2(mixed_path, noise_path, denoisy_path)
    # exec_cancellation_3(mixed_path, noise_path, denoisy_path)

    # clean_audio(audio_path, "audios/test/clean_Recording (13)_cropped.wav")

    # apply_wiener_filter(
    #     "audios/test/tts_output_playback.wav",
    #     "audios/test/noise.wav",
    #     "audios/test/wiener_2_test_3.wav",
    # )
    # wiener_filter_2(noise_path, tts_output, "audios/test/wiener_2_test_1.wav")

    # tts_saves_wav(tts_output)

    # sync 2 audio tracks :
    # command = syncstart audios/test/tts_output_padded.wav audios/test/mixed_audio_2_mono.wav
