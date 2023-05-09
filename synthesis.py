import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step), map_location=t.device('cpu'))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def synthesis(text, args):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = t.LongTensor(text).unsqueeze(0)
    text = text.cpu()  # change to cpu
    mel_input = t.zeros([1, 1, 80]).cpu()  # change to cpu
    pos_text = t.arange(1, text.size(1) + 1).unsqueeze(0)
    pos_text = pos_text.cpu()  # change to cpu

    m = m.cpu()  # change to cpu
    m_post = m_post.cpu()  # change to cpu
    m.train(False)
    m_post.train(False)

    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1, mel_input.size(1) + 1).unsqueeze(0).cpu()  # change to cpu
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            mel_input = t.cat([mel_input, mel_pred[:, -1:, :]], dim=1)

        mag_pred = m_post.forward(postnet_pred)

    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/test.wav", hp.sr, wav)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=172000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=100000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=1000)

    args = parser.parse_args()
    synthesis("The spectrogram2wav function generates a waveform from a magnitude spectrogram using the Griffin-Lim algorithm, which is an iterative phase recovery algorithm. It first normalizes the magnitude spectrogram by clipping values between 0 and 1 and scaling to the dynamic range of the spectrogram. It then converts the normalized magnitude spectrogram to the amplitude spectrogram and performs the inverse Short-Time Fourier Transform using the Griffin-Lim algorithm to obtain the waveform. Finally, it applies a de-emphasis filter to the waveform and trims any leading or trailing silence.", args)
