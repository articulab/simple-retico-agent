import audioop
import glob
import os
import wave


def downsampleWav(src, dst, inrate=44100, outrate=16000, inchannels=2, outchannels=1):
    if not os.path.exists(src):
        print("Source not found!")
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        s_read = wave.open(src, "r")
        s_write = wave.open(dst, "wb")
    except:
        print("Failed to open files!")
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1:
            converted = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print("Failed to downsample wav")
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, "NONE", "Uncompressed"))
        s_write.writeframes(converted)
    except:
        print("Failed to write wav")
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print("Failed to close wav files")
        return False

    return True


from pydub import AudioSegment as am


src = "./audios/stereo/48k/"
dest = "./audios/mono/44k/"
inrate = 48000
# outrate = 44100
outrate = 48000
inchannel = 2
outchannel = 1
# for filepath in [f for f in glob.glob(src + "*.wav")]:
#     filename = filepath.split("\\")[-1]
#     print(filename)
#     dest_path = dest + filename
#     # downsampleWav(filepath, dest+filename, inrate=96000, outrate=16000, inchannels=2, outchannels=2)
#     sound = am.from_file(filepath, format="wav", frame_rate=inrate)
#     # sound = sound.resample(sample_rate_Hz=outrate, sample_width=2, channels=1)
#     sound = sound.set_channels(outchannel)
#     sound = sound.set_frame_rate(outrate)
#     sound.export(dest_path, format="wav")


filepath = "./audios/test/Recording (12)_cropped.wav"
dest_path = "./audios/mono/Recording (12)_cropped_mono.wav"
sound = am.from_file(filepath, format="wav", frame_rate=inrate)
# sound = sound.resample(sample_rate_Hz=outrate, sample_width=2, channels=1)
sound = sound.set_channels(outchannel)
sound = sound.set_frame_rate(outrate)
sound.export(dest_path, format="wav")
