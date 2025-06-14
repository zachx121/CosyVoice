"""

"""

import logging
import os.path

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
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import time
import base64
import soundfile as sf
import torchaudio
import logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class GSVModel:
    sid_info = {
        "fuhang_zs": {"audio_fp": "/root/autodl-fs/audio_samples/fuhang/sentence1.wav",
                      "text": "我一年前打车的时候儿，我坐在副驾驶我想放平了睡会儿觉，那个司机一直放歌儿，我说师傅能不能把歌儿关了我想睡会儿觉"},
        "css_zs": {"audio_fp": "/root/autodl-fs/audio_samples/css/sentence1.wav",
                   "text": "本次航班的全体机组成员向您致以最诚挚的问候"},
        "xr1_zs": {"audio_fp":"/root/autodl-fs/audio_samples/xr_sliced/vocals4.wav",
                  "text":"他给了我一个别的电话，但是我那个电话始终打不通，五三零三九七七"},
        # "xr2_zs": {"audio_fp": "/root/autodl-fs/audio_samples/xr_sliced/vocals12.wav",
        #           "text": "诶喂您好，那个，冯主任的电话打不通呀，您这里能联系到冯主任吗？"}
    }

    def __init__(self):
        self.model = CosyVoice2('pretrained_models/CosyVoice2-0.5B',
                                load_jit=False, load_trt=False,
                                load_vllm=False,
                                fp16=False)
        for sid, value in self.sid_info.items():
            prompt_speech_16k = load_wav(value["audio_fp"], 16000)
            self.sid_info[sid]["prompt_speech_16k"] = prompt_speech_16k
            self.model.add_zero_shot_spk(prompt_text=value["text"],
                                         prompt_speech_16k=prompt_speech_16k,
                                         zero_shot_spk_id=sid)

            # self.model.save_spkinfo()
        self.sid = None

    @staticmethod
    def is_base64_audio(s):
        try:
            if len(s) % 4 != 0:
                return False
            decoded = base64.b64decode(s, validate=True)
            return len(decoded) % 2 == 0 and len(decoded) > 0
        except:
            return False

    def assign_sid(self, sid):
        if sid in self.sid_info:
            self.sid = sid
            return True
        else:
            logger.warning(f"sid('{sid}')不存在")
            return False

    def add_sid(self, sid, prompt_text, prompt_speech_16k):
        if isinstance(prompt_speech_16k, np.ndarray):
            print("numpy array")
            raise NotImplementedError("ndarray not support yet")
        elif isinstance(prompt_speech_16k, str):
            if str(prompt_speech_16k).startswith("http"):
                print("url, will download...")
                raise NotImplementedError("url not support yet")
            elif self.is_base64_audio(prompt_speech_16k):
                print("bas64, decoding...")
                audio_bytes = base64.b64decode(prompt_speech_16k)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                prompt_speech_16k = torch.from_numpy(audio_array).unsqueeze(0)
            else:
                assert os.path.exists(prompt_speech_16k), f"path not exist ('{prompt_speech_16k}')"
                prompt_speech_16k = load_wav(prompt_speech_16k, 16000)
        elif isinstance(prompt_speech_16k, torch.Tensor):
            print("torch tensor")
            pass

        assert isinstance(prompt_speech_16k, torch.Tensor), "prompt_speech_16k type error"
        self.model.add_zero_shot_spk(prompt_text=prompt_text,
                                     prompt_speech_16k=prompt_speech_16k,
                                     zero_shot_spk_id=sid)
        self.sid_info[sid] = {"text": prompt_text, "prompt_speech_16k": prompt_speech_16k}

    def format(self, audio_tensor, opt_type="int16_16k"):
        if opt_type == "int16_16k":
            res = torchaudio.transforms.Resample(orig_freq=self.model.sample_rate, new_freq=16000)(audio_tensor)
            res = res.numpy()
            # audio_arr int16 16k
            audio_arr_int16_16k = (np.clip(res, -1.0, 1.0) * 32767).astype(np.int16)[0]
            return audio_arr_int16_16k, 16000
        else:
            audio_arr_float_24k, sr = audio_tensor.numpy(), self.model.sample_rate
            return audio_arr_float_24k, sr

    def predict(self, text, sid, **kwargs):
        assert sid in self.sid_info, f"sid('{sid}')不存在"
        res = self.model.inference_cross_lingual(tts_text=text,
                                                 prompt_speech_16k=self.sid_info[sid]["prompt_speech_16k"],
                                                 zero_shot_spk_id=sid,
                                                 stream=False,
                                                 **kwargs)
        res = list(res)[0]['tts_speech']
        return self.format(res, kwargs.get("opt_type", "int16_16k"))

    # 如果调用过predict函数，需要重新执行一遍add_sid（predict调用的时跨语言合成，改操作会导致sid内部字典被改变）
    def predict_same_lang(self, text, sid, **kwargs):
        assert sid in self.sid_info, f"sid('{sid}')不存在"
        res = self.model.inference_zero_shot(tts_text=text,
                                             prompt_text=self.sid_info[sid]["text"],
                                             prompt_speech_16k=self.sid_info[sid]["prompt_speech_16k"],
                                             zero_shot_spk_id=sid,
                                             stream=False,
                                             **kwargs)
        res = list(res)[0]['tts_speech']
        return self.format(res, kwargs.get("opt_type", "int16_16k"))

    def predict_instruct(self, text, instruct_text, sid, **kwargs):
        assert sid in self.sid_info, f"sid('{sid}')不存在"
        res = self.model.inference_instruct2(tts_text=text,
                                             instruct_text=instruct_text,
                                             prompt_speech_16k=self.sid_info[sid]["prompt_speech_16k"],
                                             stream=False,
                                             **kwargs)
        res = list(res)[0]['tts_speech']
        return self.format(res, kwargs.get("opt_type", "int16_16k"))

    def predict_vc(self, source_speech_16k, sid, **kwargs):
        if isinstance(source_speech_16k, torch.Tensor):
            pass
        elif isinstance(source_speech_16k, str):
            fp = str(source_speech_16k)
            assert os.path.exists(fp)
            source_speech_16k = load_wav(fp, 16000)
        assert isinstance(source_speech_16k, torch.Tensor)

        res = self.model.inference_vc(source_speech_16k=source_speech_16k,
                                      prompt_speech_16k=self.sid_info[sid]["prompt_speech_16k"])
        res = list(res)[0]['tts_speech']
        return self.format(res, kwargs.get("opt_type", "int16_16k"))


# 模拟Notebook里的Audio函数（直接播放变成存为当前目录的wav）
def Audio(audio, rate, save_dir="audio_output"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    opt_fp = os.path.join(save_dir, f"audio_{int(time.time()*1000)}.wav")
    sf.write(opt_fp, audio, rate)
    print(f"Saved at '{opt_fp}'")


if __name__ == '__main__':
    M = GSVModel()

    audio, sr = M.predict("我是思考者", sid="css_zs")
    Audio(audio, rate=sr)
    audio, sr = M.predict_same_lang("我是思考者", sid="css_zs")
    Audio(audio, rate=sr)
    audio, sr = M.predict_instruct("我是思考者", instruct_text="用武汉话说", sid="css_zs")
    Audio(audio, rate=sr)
    audio, sr = M.predict_instruct("我是思考者", instruct_text="非常生气地说", sid="css_zs")
    Audio(audio, rate=sr)
    audio, sr = M.predict_instruct("我是思考者", instruct_text="非常高兴地说", sid="css_zs")
    Audio(audio, rate=sr)

    M.add_sid(sid="xr_abc",
              prompt_text="诶喂您好，那个，冯主任的电话打不通呀，您这里能联系到冯主任吗？",
              prompt_speech_16k="/root/autodl-fs/audio_samples/xr_sliced/vocals12.wav")
    audio, sr = M.predict("I'm not very sure about this one", sid="xr_abc")
    Audio(audio, rate=sr)
    audio, sr = M.predict_instruct("我是思考者", instruct_text="非常生气地说", sid="xr_abc")
    Audio(audio, rate=sr)
    audio, sr = M.predict_instruct("I'm thinking about this", instruct_text="非常高兴地说", sid="xr_abc")
    Audio(audio, rate=sr)

