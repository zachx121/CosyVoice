import json
import multiprocessing as mp
from flask import Flask, request, jsonify
import logging
import signal
import sys
from typing import Dict, List
import time
import pika
import torch
import base64
from datetime import datetime
from GSV_model import GSVModel

# 全局字典，存储模型信息
M_dict: Dict[str, Dict] = {}

# 存储进程信息
process_dict: Dict[str, List[mp.Process]] = {}

# 设置日志
logging.basicConfig(format='[%(asctime)s-%(name)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

rabbitmq_config = {
    "address": "120.24.144.127",
    "ports": [5672, 5673, 5674],
    "username": "admin",
    "password": "aibeeo",
    "virtual_host": "test-0208",
    # "virtual_host": "device-public"
}
queue_service_inference_request_prefix = 'queue_service_inference_request_'
exchange_service_load_model_result = 'exchange_service_load_model_result'


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


app = Flask(__name__)


def worker_process(sid: str, prompt_text: str, prompt_speech_16k: str, process_id: int):
    """子进程工作函数"""
    try:
        logger.info(f"Starting worker process {process_id} for sid: {sid}")

        # ############
        # 模型初始化
        # ############
        M = GSVModel()
        M.add_sid(sid, prompt_text, prompt_speech_16k)
        _ = M.predict("初始化推理", sid)

        # ############
        # 连接队列
        # ############
        connection, channel = connect_to_rabbitmq()
        if not connection or not channel:
            logger.error("连接mq失败")
            return  # 如果连接失败，退出函数
        # 声明队列并设置参数 队列由调用方自行创建
        # channel.queue_declare(queue=queue_service_inference_request_prefix + sid,
        #                       durable=False,  # 队列是否持久化
        #                       exclusive=False,  # 是否为独占队列
        #                       auto_delete=False,  # 是否自动删除
        #                       arguments={'x-expires': 60 * 60 * 1000})  # 额外的参数（如过期时间）

        # ############
        # 注册结束信号
        # ############
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

        # ####################
        # 模型加载成功，发送给队列
        # ####################
        load_result_event = {
            "uniqueVoiceName": sid,  # 唯一语音名称
            "loadStatus": True
        }
        channel.basic_publish(exchange=exchange_service_load_model_result, routing_key='',
                              body=json.dumps(load_result_event),
                              properties=PROPERTIES)

        logger.info(f"Worker process {process_id} initialized successfully")

        # ########################
        # 开始消费来自指定队列的数据
        # ########################
        def call_back_func(ch, method, properties, body):
            # 将字节串解码为字符串
            body_str = body.decode('utf-8')
            # 将字符串解析为字典
            info_dict = json.loads(body_str)
            trace_id = info_dict["trace_id"]
            text = info_dict["text"]
            result_queue_name = info_dict["result_queue_name"]
            try:
                audio_arr, sr = M.predict(text, sid)
                rsp = {"trace_id": trace_id,
                       "text": text,
                       "audio_buffer_int16": base64.b64encode(audio_arr.tobytes()).decode("utf-8"),
                       "sample_rate": sr}
                rsp = json.dumps({"code": 0,
                                  "msg": "",
                                  "result": rsp})
                channel.basic_publish(exchange='',
                                      routing_key=result_queue_name,
                                      body=rsp,
                                      properties=PROPERTIES)
            except Exception as e:
                rsp = json.dumps({"code": 1,
                                  "msg": f"Prediction failed, internal err {repr(e)}",
                                  "result": {}})
                channel.basic_publish(exchange='',
                                      routing_key=result_queue_name,
                                      body=rsp,
                                      properties=PROPERTIES)

        channel.basic_consume(queue=queue_service_inference_request_prefix + sid,
                              auto_ack=True,
                              on_message_callback=call_back_func)
        channel.start_consuming()
    except Exception as e:
        raise e
        logger.error(f"Fatal error in worker process {process_id}: {str(e)}")
    finally:
        logger.info(f"Worker process {process_id} shutting down")


def start_worker_processes(sid: str, prompt_text: str, prompt_speech_16k: str, num: int) -> List[mp.Process]:
    """启动指定数量的工作进程"""
    processes = []

    for i in range(num):
        process = mp.Process(
            target=worker_process,
            args=(sid, prompt_text, prompt_speech_16k, i),
            name=f"GSV_Worker_{sid}_{i}"
        )
        process.start()
        processes.append(process)
        logger.info(f"Started worker process {i} for sid: {sid}")

    return processes


def stop_processes_for_sid(sid: str):
    """停止指定sid的所有进程"""
    if sid in process_dict:
        logger.info(f"Stopping processes for sid: {sid}")

        for process in process_dict[sid]:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)  # 等待5秒

                if process.is_alive():
                    logger.warning(f"Force killing process {process.name}")
                    process.kill()
                    process.join()

        del process_dict[sid]
        logger.info(f"All processes for sid {sid} have been stopped")


@app.route('/load_model', methods=['POST'])
def load_model():
    """加载模型接口"""
    try:
        # 解析JSON参数
        data = request.get_json()

        # 验证必需参数
        required_params = ['sid', 'prompt_text', 'prompt_speech_16k', 'num']
        missing_params = [param for param in required_params if param not in data]

        if missing_params:
            return jsonify({'code': 1,
                            'msg': f'Missing required parameters: {missing_params}',
                            'result': {}}), 200

        sid = data['sid']
        prompt_text = data['prompt_text']
        prompt_speech_16k = data['prompt_speech_16k']
        num = data['num']

        logger.info(f"Loading model for sid: {sid} with {num} processes")

        if sid in M_dict:
            logger.info(f"Sid {sid} already exists")
            # stop_processes_for_sid(sid)
            return jsonify({'code': 1,
                            'msg': f'sid {sid} already exists',
                            'result': {}}), 200
        else:
            # 启动新的工作进程
            processes = start_worker_processes(sid, prompt_text, prompt_speech_16k, num)

            # 更新全局字典
            M_dict[sid] = {
                'prompt_text': prompt_text,
                'prompt_speech_16k': prompt_speech_16k,
                'num': num,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # 存储进程信息
            process_dict[sid] = processes

            logger.info(f"Successfully loaded model for sid: {sid}")

            return jsonify({'code': 0,
                            'msg': f'sid {sid} loaded',
                            'result': {}}), 200
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        return jsonify({'code': 1,
                        'msg': f'load model error. {str(e)}',
                        'result': {}}), 500


@app.route('/get_models', methods=['GET'])
def get_models():
    """获取所有已加载的模型信息"""
    res = {}
    for sid, info in M_dict.items():
        res[sid] = {k: v for k, v in info.items() if k != "prompt_speech_16k"}
    return jsonify({'code': 0,
                    'msg': '',
                    'result': res}), 200


@app.route('/unload_model', methods=['POST'])
def unload_model():
    """卸载指定的模型"""
    try:
        data = request.get_json()
        sid = data['sid']

        if sid not in M_dict:
            return jsonify({'code': 0,
                            'msg': f'Model not exist (sid: {sid})',
                            'result': {}}), 200
        # 停止进程
        stop_processes_for_sid(sid)
        # 从全局字典中删除
        del M_dict[sid]

        logger.info(f"Successfully unloaded model for sid: {sid}")

        return jsonify({'code': 0,
                        'msg': f'Model unloaded successfully for sid: {sid}',
                        'result': {}}), 200

    except Exception as e:
        logger.error(f"Error in unload_model: {str(e)}")
        return jsonify({'code': 1,
                        'msg': f'Internal server error: {str(e)}',
                        'result': {}}), 500


def cleanup_processes():
    """清理所有进程"""
    logger.info("Cleaning up all processes...")

    for sid in list(process_dict.keys()):
        stop_processes_for_sid(sid)

    M_dict.clear()
    logger.info("All processes cleaned up")


if __name__ == '__main__':
    try:
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)

        logger.info("Starting GSV Service...")
        app.run(host='0.0.0.0', port=6006, debug=False)

    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {str(e)}")
    finally:
        cleanup_processes()
