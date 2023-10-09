import argparse
from pathlib import Path

import numpy as np
import torch
import librosa
from tqdm import tqdm
import sys
sys.path.append(".")
from models.SoundNet import SoundNet
# Script global configs
LEN_WAVEFORM = 22050 * 20

local_config = {
	'batch_size': 1,
	'eps': 1e-5,
	'sample_rate': 22050,
	'load_size': 22050 * 20,
	'name_scope': 'SoundNet_TF',
	'phase': 'extract',
}
def load_audio(audio_path, sample_rate=22050, mono=True):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
    
    assert sample_rate == sr

    return sound_sample, sr

def gen_audio_from_dir(dir, file_ext='.wav', config=local_config):
    '''Audio loader from dir generator'''
    txt_list = []
    
    audio_path_list = Path(dir).glob(f'*{file_ext}')

    for audio_path in tqdm(audio_path_list):
        sound_sample, _ = load_audio(audio_path)
        yield preprocess(sound_sample, config), audio_path 

def preprocess(raw_audio, config=local_config):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]
  
    # # Scale the audio to the desired range
    # min_val = np.min(raw_audio)
    # max_val = np.max(raw_audio)
    # if max_val == min_val:
    #     raw_audio = np.zeros_like(raw_audio)
    # else:
    #     # Scale to [0, 1]
    #     raw_audio = (raw_audio - min_val) / (max_val - min_val)
    #     #  Scale to [-256, 256]  
    #     raw_audio *= 512
    #     raw_audio -= 256 

    # Scale the audio to the desired range
    raw_audio = np.clip(raw_audio * 256.0, -256, 256)

    # Make minimum length available
    length = config['load_size']
    if length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(length/raw_audio.shape[0] + 1))

    # Make equal training length
    if config['phase'] != 'extract':
        raw_audio = raw_audio[:length]

    assert len(raw_audio.shape) == 1, "Audio is not mono"
    assert np.max(raw_audio) <= 256, "Audio max value beyond 256"
    assert np.min(raw_audio) >= -256, "Audio min value beyond -256"

    # Shape for network is 1 x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [1, 1, -1, 1])

    return raw_audio.copy()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='weights/sound8.npy',
                        help='Path to the .npy file with the SoundNet weights')

    parser.add_argument('--input_dir', type=str, default='mp3',
                        help='Directory with the audio files to extract SoundNet feats')

    parser.add_argument('--output_dir', type=str, default='snf',
                        help='Dir where the audio features will be stored')
    
    parser.add_argument('--file_ext', type=str, default='.mp3',
                        help='File extension of the audio files')
    parser.add_argument("--feat_layer",type=str,default="",
                        help="the layer of feat you want to extract for later classification")

    return parser.parse_args()


def extract_features(args):
    
    model = SoundNet()
    model.load_weights(args.model_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    error_count = 0
    for sound_sample, audio_path in gen_audio_from_dir(args.input_dir,
                                                       config=local_config,
                                                       file_ext=args.file_ext):
        if sound_sample is None:
            error_count += 1
            continue
        # print(audio_path, sound_sample.shape)
        new_sample = torch.from_numpy(sound_sample)
        if torch.cuda.is_available():
            new_sample = new_sample.cuda()
        feats = model.forward(new_sample)
        # sel_feats = feats[args.feat_layer].squeeze().mean(1).cpu().data.numpy()


        # Save conv5 features
        conv5_feats = feats['conv5'].squeeze().mean(1).cpu().data.numpy()
        conv5_output_dir = Path(args.output_dir, 'conv5')
        conv5_output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(conv5_output_dir / f'{Path(audio_path).stem}.csv', conv5_feats, delimiter=';')

        # Save pool5 features
        pool5_feats = feats['pool5'].squeeze().mean(1).cpu().data.numpy()
        pool5_output_dir = Path(args.output_dir, 'pool5')
        pool5_output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(pool5_output_dir / f'{Path(audio_path).stem}.csv', pool5_feats, delimiter=';')
        
        # Save conv6 features
        conv6_feats = feats['conv6'].squeeze().mean(1).cpu().data.numpy()
        conv6_output_dir = Path(args.output_dir, 'conv6')
        conv6_output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(conv6_output_dir / f'{Path(audio_path).stem}.csv', conv6_feats, delimiter=';')

        # np.savetxt(Path(args.output_dir, f'{Path(audio_path).stem}.csv'),
        #            sel_feats,
        #            delimiter=';')

    if error_count > 0:
        print(f'Could not process {error_count} audio files correctly.')


if __name__ == '__main__':
    args = parse_args()
    extract_features(args)