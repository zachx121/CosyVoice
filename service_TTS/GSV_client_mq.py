import time
import multiprocessing
import pika
import json
import random
from typing import Dict, Any
import requests
import json
import librosa
import soundfile as sf
import numpy as np
import base64
import logging
logging.basicConfig(format='[%(asctime)s-%(name)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('pika').setLevel(logging.ERROR)

# RabbitMQ配置
rabbitmq_config = {
    "address": "120.24.144.127",
    "ports": [5672, 5673, 5674],
    "username": "admin",
    "password": "aibeeo",
    "virtual_host": "test-0208",
}
# 队列名称前缀
req_queue_prefix = "queue_service_inference_request_"
rsp_queue_prefix = "queue_service_inference_response_"
exchange_model_queue = "exchange_service_load_model_result"

def connect_to_rabbitmq():
    try:
        # 连接到 RabbitMQ
        credentials = pika.PlainCredentials(rabbitmq_config["username"], rabbitmq_config["password"])
        parameters = pika.ConnectionParameters(
            host=rabbitmq_config["address"],
            port=rabbitmq_config["ports"][0],  # 默认使用第一个端口
            virtual_host=rabbitmq_config["virtual_host"],
            credentials=credentials,
            connection_attempts=3,  # 最多尝试 3 次
            retry_delay=5,         # 每次重试间隔 5 秒
            socket_timeout=10      # 套接字超时时间为 10 秒
        )
        logger.debug("mq配置完毕，开始blocking connect连接")
        connection = pika.BlockingConnection(parameters)
        logger.debug("mq连接完毕，获取到connection")
        channel = connection.channel()
        logger.debug("mq连接完毕，获取到chanel")

        logger.info("Connected to RabbitMQ successfully.")
        # 全局消息属性
        global PROPERTIES
        PROPERTIES = pika.BasicProperties(content_type='application/json')  # 设置 content_type 为 JSON
        return connection, channel
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {repr(e)}")
        return None, None


def load_model(url="https://u212392-a844-13ff2d34.bjb1.seetacloud.com:8443",
               sid="doctor_who_0",
               prompt_text="Oh, would you? well, maybe, maybe you will win!",
               audio_fp="/Users/zhou/0-Codes/VoiceSamples/doctor_who_0.wav"):
    # 初始化mq
    req_queue = req_queue_prefix+sid
    rsp_queue = rsp_queue_prefix+sid
    connection, channel = connect_to_rabbitmq()
    channel.queue_declare(queue=req_queue,
                          durable=False,  # 队列是否持久化
                          exclusive=False,  # 是否为独占队列
                          auto_delete=False,  # 是否自动删除
                          arguments={'x-expires': 60 * 60 * 1000})  # 额外的参数（如过期时间）
    channel.queue_declare(queue=rsp_queue,
                          durable=False,  # 队列是否持久化
                          exclusive=False,  # 是否为独占队列
                          auto_delete=False,  # 是否自动删除
                          arguments={'x-expires': 60 * 60 * 1000})  # 额外的参数（如过期时间）
    channel.queue_declare(queue=exchange_model_queue,
                          durable=False,  # 队列是否持久化
                          exclusive=False,  # 是否为独占队列
                          auto_delete=False,  # 是否自动删除
                          arguments={'x-expires': 60 * 60 * 1000})  # 额外的参数（如过期时间）

    # 发起请求初始化模型
    audio_arr, sr = librosa.load(audio_fp, sr=16000, mono=True)
    if audio_arr.dtype == np.float32:
        audio_arr = (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16)
    audio_base64 = base64.b64encode(audio_arr.tobytes()).decode("utf-8")
    json_bdy = {"sid": sid,
                "prompt_text": prompt_text,
                "prompt_speech_16k": audio_base64,
                "num": 2}
    req = requests.post(url + "/load_model", json=json_bdy)
    req = json.loads(req.text)
    assert req["code"] == 0, f"code !=0, msg is {req['msg']}"

    # 等待异步的模型加载成功
    # while True:
    #     # 不明原因，GSV_serve里用同样的逻辑给这个mq推送消息了，但是一直收不到
    #     method, properties, body = channel.basic_get(queue=exchange_model_queue, auto_ack=True)
    #     if body:
    #         data = json.loads(body.decode('utf-8'))
    #         print(data)
    #         if data.get("uniqueVoiceName") == sid:
    #             break
    #     logger.info("load_model executing...")
    #     time.sleep(3.0)  # 避免CPU占用过高

    logger.info("sleep 20s...")
    while True:
        time.sleep(20)
        break

    logger.info("load_model done.")
    # 输出已加载模型的信息
    print(requests.get(url + "/get_models").text)


def send_text(sid="doctor_who_0", text="测试中文文本"):
    req_queue = req_queue_prefix + sid
    rsp_queue = rsp_queue_prefix + sid
    connection, channel = connect_to_rabbitmq()
    # 发送TTS请求给队列
    message = {
        "trace_id": f"tts_{time.time()*1000:.0f}",
        "text": text,
        "result_queue_name": rsp_queue,
    }
    channel.basic_publish(
        exchange='',
        routing_key=req_queue,
        body=json.dumps(message, ensure_ascii=False),
        properties=pika.BasicProperties(delivery_mode=2)
    )


def consume_tts_result(sid):
    def call_back_func(ch, method, properties, body):
        res = json.loads(body.decode('utf-8'))
        tid = res["result"]["trace_id"]
        sr = res["result"]["sample_rate"]
        text = res["result"]["text"]
        audio_b64 = res["result"]["audio_buffer_int16"]
        audio_arr = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
        sf.write(f"/Users/zhou/Downloads/opt_{text[:5]}...{tid}.wav", audio_arr, sr)

    connection, channel = connect_to_rabbitmq()
    # 消费TTS返回结果
    channel.basic_consume(queue=rsp_queue_prefix + sid,
                          auto_ack=True,
                          on_message_callback=call_back_func)
    channel.start_consuming()


def unload_model(url="", sid="doctor_who_1"):
    print(f">>> removing sid={sid}")
    print(requests.post(url+"/unload_model", json={"sid": sid}).text)
    print(">>> remaining models: ")
    print(requests.get(url + "/get_models").text)


if __name__ == "__main__":
    service_host = "https://u212392-a844-13ff2d34.bjb1.seetacloud.com:8443"
    sid = "doctor_who_1"
    prompt_audio = "/Users/zhou/0-Codes/VoiceSamples/doctor_who_0.wav"
    prompt_text = "Oh, would you? well, maybe, maybe you will win!"

    # 初始化模型
    load_model(url=service_host,
               sid=sid,
               prompt_text=prompt_text,
               audio_fp=prompt_audio)

    # 子进程在后台进行非阻塞式消费
    multiprocessing.Process(target=consume_tts_result, args=(sid,)).start()

    # 发送文本进行TTS
    for text in ["one two three, start to order!",
                 "one! two! three! start to order!",
                 "can add my wechat as john95, then we can be friends!",
                 "All cockroach coffee machines can use coffee powder.",
                 "Folks, today I'm bringing you an amazing mug! I've been using it for a while, and its heat retaining effect is really astonishing. I have to share it with you! The regular price is 80 yuan, but today in the live streaming room, it's only 59 yuan!",
                 ]:
        send_text(sid=sid, text=text)


    # 消费30s后结束
    time.sleep(30)
    unload_model(url=service_host, sid=sid)
