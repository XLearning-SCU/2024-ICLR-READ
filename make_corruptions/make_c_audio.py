import numpy as np
from pydub import AudioSegment
import soundfile as sf
import os
import torch.utils.data as data
import torch
import argparse
import collections


def gaussian_noise(audio_file, output_path, intensity):
    # load audio
    audio, sr = sf.read(audio_file)

    # calculate std
    noise_std = [.08, .12, 0.18, 0.26, 0.38][intensity - 1]

    # generate white noise
    noise = np.random.normal(0, noise_std, len(audio))

    # add
    audio_with_noise = audio + noise

    sf.write(output_path, audio_with_noise, sr)

def add_external_noise(audio_path, weather_path, output_path, intensity):
    audio = AudioSegment.from_file(audio_path)
    rain_sound = AudioSegment.from_file(weather_path)

    # adjust the length
    if len(audio) <= len(rain_sound):
        rain_sound = rain_sound[:len(audio)]
    else:
        print(len(audio), len(rain_sound))
        num_repeats = len(audio) // len(rain_sound) + 1
        rain_sound = rain_sound * num_repeats
        rain_sound = rain_sound[:len(audio)]
        print(len(audio), len(rain_sound))

    scale = [1, 2, 4, 6, 8]
    rain_sound = rain_sound.apply_gain(scale[intensity-1])

    output = audio.overlay(rain_sound)
    output.export(output_path, format="wav")

def make_dataset(dir, candi_audios):
    audios = []
    dir = os.path.expanduser(dir)
    # for name in sorted(os.listdir(dir)):
    for name in sorted(candi_audios):
        path = os.path.join(dir, name)
        # item = (path, name)
        audios.append(path)

    return audios

class DistortAudioFolder(data.Dataset):
    def __init__(self, root, candi_audio_names, corruption, weather_path, severity, save_path):
        audios = make_dataset(root, candi_audio_names)
        if len(audios) == 0:
            raise (RuntimeError("Found 0 audios in subfolders of: "))

        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.audio_paths = audios
        self.candi_audio_names = sorted(candi_audio_names)
        self.weather_path = weather_path
        self.save_path = save_path

    def __getitem__(self, index):
        save_path = os.path.join(self.save_path, self.corruption, 'severity_{}'.format(self.severity))
        
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except:
                print(save_path)
        print(self.candi_audio_names[index])
        
        if self.corruption == 'gaussian_noise':
            gaussian_noise(self.audio_paths[index], os.path.join(save_path, self.candi_audio_names[index]), self.severity)
        else:
            add_external_noise(self.audio_paths[index], os.path.join(self.weather_path, self.corruption + '.wav'), os.path.join(save_path, self.candi_audio_names[index]), self.severity)

        return 0 

    def __len__(self):
        return len(self.audio_paths)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--corruption', type=str, default='gaussian_noise', choices=['all', 'gaussian_noise', 'traffic', 'crowd', 'rain', 'thunder', 'wind'], help='Type of corruption to apply')
parser.add_argument('--severity', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Severity of corruption to apply')
parser.add_argument('--data_path', type=str, default='data_path/VGGSound/image_mulframe_test', help='Path to test data')
parser.add_argument('--save_path', type=str, default='data_path/VGGSound/image_mulframe_test-C', help='Path to store corruption data')
parser.add_argument('--weather_path', type=str, default='data_path/Audioset-20k/RawData/weather_audios/', help='Path to store corruption data')
args = parser.parse_args()

dir = args.data_path
candi_audio_names = os.listdir(dir)
tmp = sorted(candi_audio_names)

weather_path = args.weather_path

d = collections.OrderedDict()
if args.corruption == 'all':
    corruption_list = ['traffic', 'crowd', 'rain', 'thunder', 'wind']
else:
    corruption_list = [args.corruption]

for corruption in corruption_list:
    # for severity in range(1, 6):
    print('Adding the {} corruption (severity={}) to audios'.format(corruption, args.severity))
    distorted_dataset = DistortAudioFolder(
        root=args.data_path,
        candi_audio_names=candi_audio_names,
        corruption=corruption,
        weather_path=args.weather_path,
        severity=args.severity,
        save_path=args.save_path)
    distorted_dataset_loader = torch.utils.data.DataLoader(
        distorted_dataset, batch_size=12, shuffle=False, num_workers=0)
    for _ in distorted_dataset_loader:
        continue