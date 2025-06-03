import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# Audio('./asset/zero_shot_prompt.wav')
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# 中文直出
prompt_speech_16k = load_wav('/root/autodl-fs/voice_sample/css/ref_audio_default.wav', 16000)
# Audio('/root/autodl-fs/voice_sample/css/ref_audio_default.wav')
res = cosyvoice.inference_zero_shot(
    tts_text='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    prompt_text='本次航班的全体机组成员向您致以最诚挚的问候',
    prompt_speech_16k=prompt_speech_16k,
    stream=False)
for tensor in res:
    # Audio(tensor['tts_speech'].numpy(), rate=cosyvoice.sample_rate)
    pass


# 跨语言
prompt_speech_16k = load_wav('/root/autodl-fs/voice_sample/css/ref_audio_default.wav', 16000)
assert cosyvoice.add_zero_shot_spk('本次航班的全体机组成员向您致以最诚挚的问候', prompt_speech_16k, 'css_zero_shot')
cosyvoice.save_spkinfo()
for i, j in enumerate(cosyvoice.inference_cross_lingual('How are you today? is everything okay? can I do anything to help you cheer up?', prompt_speech_16k, stream=False)):
    # Audio(j['tts_speech'].numpy(), rate=cosyvoice.sample_rate)
    pass

# 指示性语气推理
for i, j in enumerate(cosyvoice.inference_instruct2('你今天是要闹哪样？', '用四川话说这句话', prompt_speech_16k, stream=False)):
    # Audio(j['tts_speech'].numpy(), rate=cosyvoice.sample_rate)
    # torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    pass


"""
一些特殊的语气标记
'additional_special_tokens': [
    '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
    '[breath]', '<strong>', '</strong>', '[noise]',
    '[laughter]', '[cough]', '[clucking]', '[accent]',
    '[quick_breath]',
    "<laughter>", "</laughter>",
    "[hissing]", "[sigh]", "[vocalized-noise]",
    "[lipsmack]", "[mn]"
"""

# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


