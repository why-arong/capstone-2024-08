import torch, torchaudio
from typing import List


def change_voice(src: str, ref: List[str]):
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')
    # path to 16kHz, single-channel, source waveform
    src_wav_path = '/content/SPK064KBSCU001M001.wav'
    # list of paths to all reference waveforms (each must be 16kHz, single-channel) from the target speaker
    ref_wav_paths = ['/content/SPK014KBSCU001F001.wav', "/content/SPK014KBSCU001F002.wav",
                     '/content/SPK014KBSCU001F003.wav', "/content/SPK014KBSCU001F004.wav",
                     '/content/SPK014KBSCU001F005.wav', "/content/SPK014KBSCU001F006.wav",
                     '/content/SPK014KBSCU001F007.wav',]

    query_seq = knn_vc.get_features(src_wav_path)
    matching_set = knn_vc.get_matching_set(ref_wav_paths)
    out_wav = knn_vc.match(query_seq, matching_set, topk=4)
    torchaudio.save("./result/vc_out.wav", out_wav[None], 16000)
