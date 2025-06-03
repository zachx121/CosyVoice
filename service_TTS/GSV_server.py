"""
[route]: /train_model
[json-params]: {speaker:"..", lang:"zh", data_urls:["...", "..." ]}

[route]: /model_status
[json-params]: {speaker_list: ["..", ".."]}


[route]: /add_reference
[json-params]:
{"speaker": "..",
 "ref_audio_url": "...",
 "ref_text_url": "...", 注意文本格式文本是 "zh_cn|你好我是四郎" 这样的，即按竖线分割的，前面是语言后面是音频的文本
 "ref_suffix":".." optional 可以直接不传，传了的话后面推理接口也要传。当可以提供多个参考音频时，可以用情绪作为后缀}

[route]: /init_model
[json-params]: {speaker:"..."}

[route]: /inference
[json-params]:
    trace_id: str = None
    speaker: str = None  # 角色音
    text: str = None  # 要合成的文本
    lang: str = None  # 合成音频的语言 (e.g. "JP", "ZH", "EN", "ZH_EN", "JP_EN")
    use_ref: bool = True  # 推理时是否使用参考音频的情绪
    ref_suffix: str = D_REF_SUFFIX  # 可不传，用于指定参考音频的后缀
"""

import re
import sys
import torch
import os
os.environ['TQDM_DISABLE'] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import base64
import json
from subprocess import getstatusoutput, check_output
from flask import Flask, request
import signal
from GSV_model import GSVModel
import socket
import pika
import traceback  # 导入 traceback 模块

# 关闭pika的INFO及以下的日志
logging.getLogger('pika').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="./static_folder", static_url_path="")

# 日志目录
log_dir = "logs"
# 日志文件名
log_file = "server.log"
# RabbitMQ 连接信息
rabbitmq_config = {
    "address": "120.24.144.127",
    "ports": [5672, 5673, 5674],
    "username": "admin",
    "password": "aibeeo",
    "virtual_host": "test-0208",
    # "virtual_host": "device-public"
}
queue_service_inference_request_prefix='queue_service_inference_request_'
exchange_service_load_model_result='exchange_service_load_model_result'


def get_machine_id():
    """获取机器的主机名，清理并返回。"""
    machine_id = None
    try:
        # 尝试通过 socket 获取主机名
        machine_id = socket.gethostname()

        # 如果获取失败，尝试通过环境变量获取
        if not machine_id:
            machine_id = os.getenv("HOSTNAME")  # Linux/Docker
        if not machine_id:
            machine_id = os.getenv("COMPUTERNAME")  # Windows

        # 清理主机名：只保留字母、数字和横线
        if machine_id:
            machine_id = re.sub(r'[^a-zA-Z0-9\-]', '', machine_id).lower()
        logger.info("get  machine ID: %s", machine_id)
        return machine_id

    except Exception as e:
        logger.error("Error getting machine ID: %s", repr(e))
        return None


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
        logger.info("mq配置完毕，开始blocking connect连接")
        connection = pika.BlockingConnection(parameters)
        logger.info("mq连接完毕，获取到connection")
        channel = connection.channel()
        logger.info("mq连接完毕，获取到chanel")

        logger.info("Connected to RabbitMQ successfully.")
        # 全局消息属性
        global PROPERTIES
        PROPERTIES = pika.BasicProperties(content_type='application/json')  # 设置 content_type 为 JSON
        return connection, channel
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {repr(e)}")
        return None, None


def model_process(sid: str):
    M = None
    # global logger
    # 连接到 RabbitMQ
    connection, channel = connect_to_rabbitmq()
    if not connection or not channel:
        logger.error("连接mq失败")
        return  # 如果连接失败，退出函数

    def signal_handler(sig, frame):
        # 关闭通道和连接
        nonlocal M
        if channel is not None:
            channel.stop_consuming()
            channel.close()
        if connection is not None:
            connection.close()
        logger.info(f"Close process of sid={sid}: Channel and connection closed.")
        if M is not None:
            del M
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        logger.info(f"Close process of sid={sid}: Model deleted.")
        sys.exit(0)

    # 注册信号处理器 处理外部主进程触发的 p.terminate()
    signal.signal(signal.SIGTERM, signal_handler)

    # 获取机器的 hostname
    logger.info("开始运行model_process")
    machine_id = get_machine_id()
    if not machine_id:
        logger.error("Failed to retrieve machine ID.")
        return
    logger.info("检测模型文件是否存在")

    request_queue_name = queue_service_inference_request_prefix+sid
    # 设置过期时间（单位：毫秒），这里设置为 1 小时（3600000 毫秒）
    args = {
        'x-expires': 60 * 60 * 1000  # 设置过期时间
    }
    # 声明队列并设置参数
    channel.queue_declare(queue=request_queue_name,
                          durable=False,    # 队列是否持久化
                          exclusive=False,  # 是否为独占队列
                          auto_delete=False,  # 是否自动删除
                          arguments=args)   # 额外的参数（如过期时间）
    M = GSVModel()

    # 预热推理 | 特意放在event之后，避免加载等太久
    _ = M.predict("Hi there, how is your day.")

    # 发送load成功事件
    logger.debug("发送load成功事件到mq")
    load_result_event = {
        "uniqueVoiceName": sid,  # 唯一语音名称
        "loadStatus": True
    }
    channel.basic_publish(exchange=exchange_service_load_model_result, routing_key='', body=json.dumps(load_result_event),  properties=PROPERTIES)

    def call_back_func(ch, method, properties, body):
        try:
            # 将字节串解码为字符串
            body_str = body.decode('utf-8')
            # 将字符串解析为字典
            info_dict = json.loads(body_str)
            tid = info_dict["trace_id"]
            res_queue = info_dict["result_queue_name"]
            text = info_dict["text"]
            wav_arr_int16_16k, sr = M.predict_format(text)
            rsp = {"trace_id": tid,
                   "audio_buffer_int16": base64.b64encode(wav_arr_int16_16k.tobytes()).decode(),
                   "sample_rate": 16000,
                   "audio_text": text
                   }
            rsp = json.dumps({"code": 0,
                              "msg": "",
                              "result": rsp})
            channel.basic_publish(exchange='', routing_key=res_queue, body=rsp, properties=PROPERTIES)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析错误: {e}")
            logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
        except Exception as e:
            logger.error(f"创建 InferenceParam 实例时出错: {e}")
            logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
            raise e

    channel.basic_consume(queue=request_queue_name, auto_ack=True, on_message_callback=call_back_func)
    try:
        channel.start_consuming()
    except Exception as e:
        logger.error(f"Error during consuming in model_process. sid={sid}, error: {e}")
        logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
    finally:
        # 关闭通道和连接
        logger.warning(f"close channel/connection and del M in try-catch...")
        channel.close()
        connection.close()
        del M
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# 返回所有GPU的内存空余量，是一个list
def get_free_gpu_mem():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


# python -m service_GSV.GSV_server  # 由于用到了相对路径的import，必须以module形式执行
if __name__ == '__main__':
    model_process(sid="fuhang_zs")

