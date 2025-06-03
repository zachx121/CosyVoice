
import logging

#a
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('jieba_fast').setLevel(logging.WARNING)
logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

import numpy as np
import torch
import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('./')
import cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


class GSVModel:
    sid_info = {
        "fuhang_zs": {"audio_fp": "/root/autodl-fs/audio_samples/fuhang/sentence1.wav",
                      "text": "我一年前打车的时候儿，我坐在副驾驶我想放平了睡会儿觉，那个司机一直放歌儿，我说师傅能不能把歌儿关了我想睡会儿觉"},
        "css_zs": {"audio_fp": "/root/autodl-fs/audio_samples/css/sentence1.wav",
                   "text": "本次航班的全体机组成员向您致以最诚挚的问候"},
        "xr1_zs": {"audio_fp":"/root/autodl-fs/audio_samples/xr_sliced/vocals4.wav",
                  "text":"他给了我一个别的电话，但是我那个电话始终打不通，五三零三九七七"},
        "xr2_zs": {"audio_fp": "/root/autodl-fs/audio_samples/xr_sliced/vocals12.wav",
                  "text": "诶喂您好，那个，冯主任的电话打不通呀，您这里能联系到冯主任吗？"}
    }

    def __init__(self, sid="fuhang_zs"):
        self.model = CosyVoice2('pretrained_models/CosyVoice2-0.5B',
                                load_jit=False, load_trt=False,
                                load_vllm=False,
                                fp16=False)
        self.sid = sid
        self.prompt_speech_16k = load_wav(self.sid_info[self.sid]["audio_fp"], 16000)
        assert self.model.add_zero_shot_spk(self.sid_info[self.sid]["text"], self.prompt_speech_16k, self.sid)
        self.model.save_spkinfo()

    def predict(self, text, **kwargs):
        self.model.add_zero_shot_spk(self.sid_info[self.sid]["text"], self.prompt_speech_16k, self.sid)
        res = self.model.inference_cross_lingual(text,
                                                 self.prompt_speech_16k,
                                                 stream=False,
                                                 zero_shot_spk_id=self.sid,
                                                 **kwargs)
        audio_arr_float_24k, sr = list(res)[0]['tts_speech'].numpy(), self.model.sample_rate
        return audio_arr_float_24k, sr

    def predict_format(self, text, **kwargs):
        self.model.add_zero_shot_spk(self.sid_info[self.sid]["text"], self.prompt_speech_16k, self.sid)
        res = self.model.inference_cross_lingual(text,
                                                 self.prompt_speech_16k,
                                                 stream=False,
                                                 zero_shot_spk_id=self.sid,
                                                 **kwargs)
        # tensor
        res = list(res)[0]['tts_speech']
        # audio_arr float32 16k
        res = torchaudio.transforms.Resample(orig_freq=self.model.sample_rate, new_freq=16000)(res)
        res = res.numpy()
        # audio_arr int16 16k
        res = (np.clip(res, -1.0, 1.0) * 32767).astype(np.int16)[0]
        return res, 16000

