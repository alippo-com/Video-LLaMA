"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_imgorvideo(gr_video, gr_img, text_input, chat_state,chatbot,audio_flag):
    print("%s | %s | %s | %s | %s | %s" % (gr_video, gr_img, text_input, chat_state, chatbot, audio_flag))
    if chatbot is None:
        chatbot = []

    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
             chat_state = conv_llava_llama_2.copy()
    if gr_img is None and gr_video is None:
        return None, None, None, gr.update(interactive=True), chat_state, None
    elif gr_img is not None and gr_video is None:
        print(gr_img)
        chatbot = chatbot + [((gr_img,), None)]
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    elif gr_video is not None and gr_img is None:
        print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        chat_state.system =  ""
        img_list = []
        if audio_flag:
            llm_message = chat.upload_video(gr_video, chat_state, img_list)
        else:
            llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    else:
        # img_list = []
        return gr.update(interactive=False), gr.update(interactive=False, placeholder='Currently, only one input is supported'), gr.update(value="Currently, only one input is supported", interactive=False), chat_state, None,chatbot

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state, img_list

title = """
<h1 align="center"><a href="https://github.com/DAMO-NLP-SG/Video-LLaMA"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTA0IiBoZWlnaHQ9IjMzIiB2aWV3Qm94PSIwIDAgMTA0IDMzIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNMC44NiAxNi43QzAuODYgMTUuMzggMS4xMDY2NyAxNC4xNCAxLjYgMTIuOThDMi4xMDY2NyAxMS44MiAyLjc5MzMzIDEwLjgxMzMgMy42NiA5Ljk2QzQuNTI2NjcgOS4wOTMzMyA1LjU0IDguNDEzMzMgNi43IDcuOTJDNy44NiA3LjQxMzMzIDkuMDkzMzMgNy4xNiAxMC40IDcuMTZDMTEuNzA2NyA3LjE2IDEyLjkzMzMgNy40MDY2NyAxNC4wOCA3LjlDMTUuMjI2NyA4LjM5MzMzIDE2LjIyNjcgOS4wNjY2NyAxNy4wOCA5LjkyQzE3Ljk0NjcgMTAuNzYgMTguNjI2NyAxMS43NTMzIDE5LjEyIDEyLjlDMTkuNjI2NyAxNC4wNDY3IDE5Ljg5MzMgMTUuMjczMyAxOS45MiAxNi41OFYyNC40NEMxOS45MiAyNC44NjY3IDE5Ljc3MzMgMjUuMjMzMyAxOS40OCAyNS41NEMxOS4xODY3IDI1Ljg0NjcgMTguODIgMjYgMTguMzggMjZDMTcuOTUzMyAyNiAxNy42IDI1Ljg0NjcgMTcuMzIgMjUuNTRDMTcuMDUzMyAyNS4yMzMzIDE2LjkyIDI0Ljg2NjcgMTYuOTIgMjQuNDRMMTYuOSAyMy40NEMxNi4wNDY3IDI0LjI4IDE1LjA2NjcgMjQuOTYgMTMuOTYgMjUuNDhDMTIuODUzMyAyNS45ODY3IDExLjY2NjcgMjYuMjQgMTAuNCAyNi4yNEM5LjA5MzMzIDI2LjI0IDcuODYgMjUuOTkzMyA2LjcgMjUuNUM1LjU0IDI0Ljk5MzMgNC41MjY2NyAyNC4zMDY3IDMuNjYgMjMuNDRDMi43OTMzMyAyMi41NzMzIDIuMTA2NjcgMjEuNTY2NyAxLjYgMjAuNDJDMS4xMDY2NyAxOS4yNiAwLjg2IDE4LjAyIDAuODYgMTYuN1pNMy44OCAxNi43QzMuODggMTcuNTkzMyA0LjA0NjY3IDE4LjQzMzMgNC4zOCAxOS4yMkM0LjcyNjY3IDIwLjAwNjcgNS4xOTMzMyAyMC43IDUuNzggMjEuM0M2LjM2NjY3IDIxLjg4NjcgNy4wNTMzMyAyMi4zNTMzIDcuODQgMjIuN0M4LjY0IDIzLjA0NjcgOS40OTMzMyAyMy4yMiAxMC40IDIzLjIyQzExLjMyIDIzLjIyIDEyLjE3MzMgMjMuMDQ2NyAxMi45NiAyMi43QzEzLjc2IDIyLjM1MzMgMTQuNDUzMyAyMS44ODY3IDE1LjA0IDIxLjNDMTUuNjI2NyAyMC43IDE2LjA4NjcgMjAuMDA2NyAxNi40MiAxOS4yMkMxNi43NTMzIDE4LjQzMzMgMTYuOTIgMTcuNTkzMyAxNi45MiAxNi43QzE2LjkyIDE1Ljc5MzMgMTYuNzUzMyAxNC45NDY3IDE2LjQyIDE0LjE2QzE2LjA4NjcgMTMuMzYgMTUuNjI2NyAxMi42NjY3IDE1LjA0IDEyLjA4QzE0LjQ1MzMgMTEuNDkzMyAxMy43NiAxMS4wMzMzIDEyLjk2IDEwLjdDMTIuMTczMyAxMC4zNTMzIDExLjMyIDEwLjE4IDEwLjQgMTAuMThDOS40OTMzMyAxMC4xOCA4LjY0IDEwLjM1MzMgNy44NCAxMC43QzcuMDUzMzMgMTEuMDMzMyA2LjM2NjY3IDExLjQ5MzMgNS43OCAxMi4wOEM1LjE5MzMzIDEyLjY2NjcgNC43MjY2NyAxMy4zNiA0LjM4IDE0LjE2QzQuMDQ2NjcgMTQuOTQ2NyAzLjg4IDE1Ljc5MzMgMy44OCAxNi43Wk0yNy43MzQxIDEuODhDMjcuNzM0MSAxLjQ1MzMzIDI3LjU4NzQgMS4xIDI3LjI5NDEgMC44MTk5OTlDMjcuMDAwNyAwLjUzOTk5OCAyNi42NDc0IDAuMzk5OTk4IDI2LjIzNDEgMC4zOTk5OThDMjUuODIwNyAwLjM5OTk5OCAyNS40Njc0IDAuNTM5OTk4IDI1LjE3NDEgMC44MTk5OTlDMjQuODgwNyAxLjEgMjQuNzM0MSAxLjQ1MzMzIDI0LjczNDEgMS44OFYyNC41MkMyNC43MzQxIDI0LjkzMzMgMjQuODgwNyAyNS4yODY3IDI1LjE3NDEgMjUuNThDMjUuNDY3NCAyNS44NiAyNS44MjA3IDI2IDI2LjIzNDEgMjZDMjYuNjQ3NCAyNiAyNy4wMDA3IDI1Ljg2IDI3LjI5NDEgMjUuNThDMjcuNTg3NCAyNS4yODY3IDI3LjczNDEgMjQuOTMzMyAyNy43MzQxIDI0LjUyVjEuODhaTTMyLjI4NTMgMi44NEMzMi4yODUzIDMuMzIgMzIuNDUyIDMuNzMzMzMgMzIuNzg1MyA0LjA4QzMzLjEzMiA0LjQyNjY3IDMzLjU1MiA0LjYgMzQuMDQ1MyA0LjZDMzQuNTM4NiA0LjYgMzQuOTUyIDQuNDI2NjcgMzUuMjg1MyA0LjA4QzM1LjYzMiAzLjczMzMzIDM1LjgwNTMgMy4zMiAzNS44MDUzIDIuODRDMzUuODA1MyAyLjM0NjY3IDM1LjYzMiAxLjkzMzMzIDM1LjI4NTMgMS42QzM0Ljk1MiAxLjI1MzMzIDM0LjUzODYgMS4wOCAzNC4wNDUzIDEuMDhDMzMuNTUyIDEuMDggMzMuMTMyIDEuMjUzMzMgMzIuNzg1MyAxLjZDMzIuNDUyIDEuOTMzMzMgMzIuMjg1MyAyLjM0NjY3IDMyLjI4NTMgMi44NFpNMzUuNTQ1MyA4LjY2QzM1LjU0NTMgOC4yMzMzMyAzNS40MDUzIDcuODczMzMgMzUuMTI1MyA3LjU4QzM0Ljg0NTMgNy4yODY2NyAzNC40OTIgNy4xNCAzNC4wNjUzIDcuMTRDMzMuNjM4NiA3LjE0IDMzLjI3ODYgNy4yODY2NyAzMi45ODUzIDcuNThDMzIuNjkyIDcuODczMzMgMzIuNTQ1MyA4LjIzMzMzIDMyLjU0NTMgOC42NlYyNC41QzMyLjU0NTMgMjQuOTEzMyAzMi42OTIgMjUuMjY2NyAzMi45ODUzIDI1LjU2QzMzLjI5MiAyNS44NTMzIDMzLjY1MiAyNiAzNC4wNjUzIDI2QzM0LjQ3ODYgMjYgMzQuODI1MyAyNS44NTMzIDM1LjEwNTMgMjUuNTZDMzUuMzk4NiAyNS4yNjY3IDM1LjU0NTMgMjQuOTEzMyAzNS41NDUzIDI0LjVWOC42NlpNNDMuMzM3NSA4LjY0QzQzLjMzNzUgOC4yMTMzMyA0My4xOTA4IDcuODYgNDIuODk3NSA3LjU4QzQyLjYwNDIgNy4zIDQyLjI1MDggNy4xNiA0MS44Mzc1IDcuMTZDNDEuNDI0MiA3LjE2IDQxLjA3MDggNy4zIDQwLjc3NzUgNy41OEM0MC40ODQyIDcuODYgNDAuMzM3NSA4LjIxMzMzIDQwLjMzNzUgOC42NFYzMS4yOEM0MC4zMzc1IDMxLjY5MzMgNDAuNDg0MiAzMi4wNCA0MC43Nzc1IDMyLjMyQzQxLjA3MDggMzIuNjEzMyA0MS40MjQyIDMyLjc2IDQxLjgzNzUgMzIuNzZDNDIuMjUwOCAzMi43NiA0Mi42MDQyIDMyLjYxMzMgNDIuODk3NSAzMi4zMkM0My4xOTA4IDMyLjA0IDQzLjMzNzUgMzEuNjkzMyA0My4zMzc1IDMxLjI4VjIzLjY0QzQ0LjE5MDggMjQuNDUzMyA0NS4xNzc1IDI1LjA5MzMgNDYuMjk3NSAyNS41NkM0Ny40MTc1IDI2LjAxMzMgNDguNjEwOCAyNi4yNCA0OS44Nzc1IDI2LjI0QzUxLjE5NzUgMjYuMjQgNTIuNDMwOCAyNS45OTMzIDUzLjU3NzUgMjUuNUM1NC43Mzc1IDI0Ljk5MzMgNTUuNzQ0MiAyNC4zMDY3IDU2LjU5NzUgMjMuNDRDNTcuNDY0MiAyMi41NzMzIDU4LjE0NDIgMjEuNTY2NyA1OC42Mzc1IDIwLjQyQzU5LjE0NDIgMTkuMjYgNTkuMzk3NSAxOC4wMiA1OS4zOTc1IDE2LjdDNTkuMzk3NSAxNS4zOCA1OS4xNDQyIDE0LjE0IDU4LjYzNzUgMTIuOThDNTguMTQ0MiAxMS44MiA1Ny40NjQyIDEwLjgxMzMgNTYuNTk3NSA5Ljk2QzU1Ljc0NDIgOS4wOTMzMyA1NC43Mzc1IDguNDEzMzMgNTMuNTc3NSA3LjkyQzUyLjQzMDggNy40MTMzMyA1MS4xOTc1IDcuMTYgNDkuODc3NSA3LjE2QzQ4LjYxMDggNy4xNiA0Ny40MTc1IDcuMzkzMzMgNDYuMjk3NSA3Ljg2QzQ1LjE3NzUgOC4zMjY2NyA0NC4xOTA4IDguOTY2NjcgNDMuMzM3NSA5Ljc4VjguNjRaTTQzLjM1NzUgMTYuN0M0My4zNTc1IDE1Ljc5MzMgNDMuNTI0MiAxNC45NDY3IDQzLjg1NzUgMTQuMTZDNDQuMjA0MiAxMy4zNiA0NC42NzA4IDEyLjY2NjcgNDUuMjU3NSAxMi4wOEM0NS44NDQyIDExLjQ5MzMgNDYuNTMwOCAxMS4wMzMzIDQ3LjMxNzUgMTAuN0M0OC4xMTc1IDEwLjM1MzMgNDguOTcwOCAxMC4xOCA0OS44Nzc1IDEwLjE4QzUwLjc5NzUgMTAuMTggNTEuNjUwOCAxMC4zNTMzIDUyLjQzNzUgMTAuN0M1My4yMzc1IDExLjAzMzMgNTMuOTMwOCAxMS40OTMzIDU0LjUxNzUgMTIuMDhDNTUuMTA0MiAxMi42NjY3IDU1LjU2NDIgMTMuMzYgNTUuODk3NSAxNC4xNkM1Ni4yMzA4IDE0Ljk0NjcgNTYuMzk3NSAxNS43OTMzIDU2LjM5NzUgMTYuN0M1Ni4zOTc1IDE3LjU5MzMgNTYuMjMwOCAxOC40MzMzIDU1Ljg5NzUgMTkuMjJDNTUuNTY0MiAyMC4wMDY3IDU1LjEwNDIgMjAuNyA1NC41MTc1IDIxLjNDNTMuOTMwOCAyMS44ODY3IDUzLjIzNzUgMjIuMzUzMyA1Mi40Mzc1IDIyLjdDNTEuNjUwOCAyMy4wNDY3IDUwLjc5NzUgMjMuMjIgNDkuODc3NSAyMy4yMkM0OC45NzA4IDIzLjIyIDQ4LjExNzUgMjMuMDQ2NyA0Ny4zMTc1IDIyLjdDNDYuNTMwOCAyMi4zNTMzIDQ1Ljg0NDIgMjEuODg2NyA0NS4yNTc1IDIxLjNDNDQuNjcwOCAyMC43IDQ0LjIwNDIgMjAuMDA2NyA0My44NTc1IDE5LjIyQzQzLjUyNDIgMTguNDMzMyA0My4zNTc1IDE3LjU5MzMgNDMuMzU3NSAxNi43Wk02Ni4wNTE2IDguNjRDNjYuMDUxNiA4LjIxMzMzIDY1LjkwNDkgNy44NiA2NS42MTE2IDcuNThDNjUuMzE4MiA3LjMgNjQuOTY0OSA3LjE2IDY0LjU1MTYgNy4xNkM2NC4xMzgyIDcuMTYgNjMuNzg0OSA3LjMgNjMuNDkxNiA3LjU4QzYzLjE5ODIgNy44NiA2My4wNTE2IDguMjEzMzMgNjMuMDUxNiA4LjY0VjMxLjI4QzYzLjA1MTYgMzEuNjkzMyA2My4xOTgyIDMyLjA0IDYzLjQ5MTYgMzIuMzJDNjMuNzg0OSAzMi42MTMzIDY0LjEzODIgMzIuNzYgNjQuNTUxNiAzMi43NkM2NC45NjQ5IDMyLjc2IDY1LjMxODIgMzIuNjEzMyA2NS42MTE2IDMyLjMyQzY1LjkwNDkgMzIuMDQgNjYuMDUxNiAzMS42OTMzIDY2LjA1MTYgMzEuMjhWMjMuNjRDNjYuOTA0OSAyNC40NTMzIDY3Ljg5MTYgMjUuMDkzMyA2OS4wMTE2IDI1LjU2QzcwLjEzMTYgMjYuMDEzMyA3MS4zMjQ5IDI2LjI0IDcyLjU5MTYgMjYuMjRDNzMuOTExNiAyNi4yNCA3NS4xNDQ5IDI1Ljk5MzMgNzYuMjkxNiAyNS41Qzc3LjQ1MTYgMjQuOTkzMyA3OC40NTgyIDI0LjMwNjcgNzkuMzExNiAyMy40NEM4MC4xNzgyIDIyLjU3MzMgODAuODU4MiAyMS41NjY3IDgxLjM1MTYgMjAuNDJDODEuODU4MiAxOS4yNiA4Mi4xMTE2IDE4LjAyIDgyLjExMTYgMTYuN0M4Mi4xMTE2IDE1LjM4IDgxLjg1ODIgMTQuMTQgODEuMzUxNiAxMi45OEM4MC44NTgyIDExLjgyIDgwLjE3ODIgMTAuODEzMyA3OS4zMTE2IDkuOTZDNzguNDU4MiA5LjA5MzMzIDc3LjQ1MTYgOC40MTMzMyA3Ni4yOTE2IDcuOTJDNzUuMTQ0OSA3LjQxMzMzIDczLjkxMTYgNy4xNiA3Mi41OTE2IDcuMTZDNzEuMzI0OSA3LjE2IDcwLjEzMTYgNy4zOTMzMyA2OS4wMTE2IDcuODZDNjcuODkxNiA4LjMyNjY3IDY2LjkwNDkgOC45NjY2NyA2Ni4wNTE2IDkuNzhWOC42NFpNNjYuMDcxNiAxNi43QzY2LjA3MTYgMTUuNzkzMyA2Ni4yMzgyIDE0Ljk0NjcgNjYuNTcxNiAxNC4xNkM2Ni45MTgyIDEzLjM2IDY3LjM4NDkgMTIuNjY2NyA2Ny45NzE2IDEyLjA4QzY4LjU1ODIgMTEuNDkzMyA2OS4yNDQ5IDExLjAzMzMgNzAuMDMxNiAxMC43QzcwLjgzMTYgMTAuMzUzMyA3MS42ODQ5IDEwLjE4IDcyLjU5MTYgMTAuMThDNzMuNTExNiAxMC4xOCA3NC4zNjQ5IDEwLjM1MzMgNzUuMTUxNiAxMC43Qzc1Ljk1MTYgMTEuMDMzMyA3Ni42NDQ5IDExLjQ5MzMgNzcuMjMxNiAxMi4wOEM3Ny44MTgyIDEyLjY2NjcgNzguMjc4MiAxMy4zNiA3OC42MTE2IDE0LjE2Qzc4Ljk0NDkgMTQuOTQ2NyA3OS4xMTE2IDE1Ljc5MzMgNzkuMTExNiAxNi43Qzc5LjExMTYgMTcuNTkzMyA3OC45NDQ5IDE4LjQzMzMgNzguNjExNiAxOS4yMkM3OC4yNzgyIDIwLjAwNjcgNzcuODE4MiAyMC43IDc3LjIzMTYgMjEuM0M3Ni42NDQ5IDIxLjg4NjcgNzUuOTUxNiAyMi4zNTMzIDc1LjE1MTYgMjIuN0M3NC4zNjQ5IDIzLjA0NjcgNzMuNTExNiAyMy4yMiA3Mi41OTE2IDIzLjIyQzcxLjY4NDkgMjMuMjIgNzAuODMxNiAyMy4wNDY3IDcwLjAzMTYgMjIuN0M2OS4yNDQ5IDIyLjM1MzMgNjguNTU4MiAyMS44ODY3IDY3Ljk3MTYgMjEuM0M2Ny4zODQ5IDIwLjcgNjYuOTE4MiAyMC4wMDY3IDY2LjU3MTYgMTkuMjJDNjYuMjM4MiAxOC40MzMzIDY2LjA3MTYgMTcuNTkzMyA2Ni4wNzE2IDE2LjdaTTg3LjY0NTYgMTYuN0M4Ny42NDU2IDE1Ljc5MzMgODcuODEyMyAxNC45NDY3IDg4LjE0NTYgMTQuMTZDODguNDkyMyAxMy4zNiA4OC45NTkgMTIuNjY2NyA4OS41NDU2IDEyLjA4QzkwLjEzMjMgMTEuNDkzMyA5MC44MTkgMTEuMDMzMyA5MS42MDU2IDEwLjdDOTIuNDA1NiAxMC4zNTMzIDkzLjI1OSAxMC4xOCA5NC4xNjU2IDEwLjE4Qzk1LjA4NTYgMTAuMTggOTUuOTM5IDEwLjM1MzMgOTYuNzI1NiAxMC43Qzk3LjUyNTYgMTEuMDMzMyA5OC4yMTkgMTEuNDkzMyA5OC44MDU2IDEyLjA4Qzk5LjM5MjMgMTIuNjY2NyA5OS44NTIzIDEzLjM2IDEwMC4xODYgMTQuMTZDMTAwLjUxOSAxNC45NDY3IDEwMC42ODYgMTUuNzkzMyAxMDAuNjg2IDE2LjdDMTAwLjY4NiAxNy41OTMzIDEwMC41MTkgMTguNDMzMyAxMDAuMTg2IDE5LjIyQzk5Ljg1MjMgMjAuMDA2NyA5OS4zOTIzIDIwLjcgOTguODA1NiAyMS4zQzk4LjIxOSAyMS44ODY3IDk3LjUyNTYgMjIuMzUzMyA5Ni43MjU2IDIyLjdDOTUuOTM5IDIzLjA0NjcgOTUuMDg1NiAyMy4yMiA5NC4xNjU2IDIzLjIyQzkzLjI1OSAyMy4yMiA5Mi40MDU2IDIzLjA0NjcgOTEuNjA1NiAyMi43QzkwLjgxOSAyMi4zNTMzIDkwLjEzMjMgMjEuODg2NyA4OS41NDU2IDIxLjNDODguOTU5IDIwLjcgODguNDkyMyAyMC4wMDY3IDg4LjE0NTYgMTkuMjJDODcuODEyMyAxOC40MzMzIDg3LjY0NTYgMTcuNTkzMyA4Ny42NDU2IDE2LjdaTTg0LjYyNTYgMTYuN0M4NC42MjU2IDE4LjAyIDg0Ljg3MjMgMTkuMjYgODUuMzY1NiAyMC40MkM4NS44NzIzIDIxLjU2NjcgODYuNTU5IDIyLjU3MzMgODcuNDI1NiAyMy40NEM4OC4yOTIzIDI0LjMwNjcgODkuMzA1NiAyNC45OTMzIDkwLjQ2NTYgMjUuNUM5MS42MjU2IDI1Ljk5MzMgOTIuODU5IDI2LjI0IDk0LjE2NTYgMjYuMjRDOTUuNDg1NiAyNi4yNCA5Ni43MTkgMjUuOTkzMyA5Ny44NjU2IDI1LjVDOTkuMDI1NiAyNC45OTMzIDEwMC4wMzIgMjQuMzA2NyAxMDAuODg2IDIzLjQ0QzEwMS43NTIgMjIuNTczMyAxMDIuNDMyIDIxLjU2NjcgMTAyLjkyNiAyMC40MkMxMDMuNDMyIDE5LjI2IDEwMy42ODYgMTguMDIgMTAzLjY4NiAxNi43QzEwMy42ODYgMTUuMzggMTAzLjQzMiAxNC4xNCAxMDIuOTI2IDEyLjk4QzEwMi40MzIgMTEuODIgMTAxLjc1MiAxMC44MTMzIDEwMC44ODYgOS45NkMxMDAuMDMyIDkuMDkzMzMgOTkuMDI1NiA4LjQxMzMzIDk3Ljg2NTYgNy45MkM5Ni43MTkgNy40MTMzMyA5NS40ODU2IDcuMTYgOTQuMTY1NiA3LjE2QzkyLjg1OSA3LjE2IDkxLjYyNTYgNy40MTMzMyA5MC40NjU2IDcuOTJDODkuMzA1NiA4LjQxMzMzIDg4LjI5MjMgOS4wOTMzMyA4Ny40MjU2IDkuOTZDODYuNTU5IDEwLjgxMzMgODUuODcyMyAxMS44MiA4NS4zNjU2IDEyLjk4Qzg0Ljg3MjMgMTQuMTQgODQuNjI1NiAxNS4zOCA4NC42MjU2IDE2LjdaIiBmaWxsPSIjRTMwMDQ3Ii8+Cjwvc3ZnPgo=", alt="Video-LLaMA" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>
<h1 align="center">Alippo: Empowering Women Enterpreneurs</h1>

<h5 align="center">  Alippo is a live upskilling platform for women to learn new skills and set up their home businesses across various categories, such as chocolate making, baking, </h5>

"""
Note_markdown = ("""
### Note
The output results may be influenced by input quality, limitations of the dataset, and the model's susceptibility to illusions. Please interpret the results with caution.

**Copyright 2023 Alippo.**
""")


cite_markdown = ("""
## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@article{damonlpsg2023videollama,
  author = {Zhang, Hang and Li, Xin and Bing, Lidong},
  title = {Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding},
  year = 2023,
  journal = {arXiv preprint arXiv:2306.02858}
  url = {https://arxiv.org/abs/2306.02858}
}
""")

case_note_upload = ("""
### We provide some examples at the bottom of the page. Simply click on them to try them out directly.
""")

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=0.5):
            video = gr.Video()
            image = gr.Image(type="filepath")
            gr.Markdown(case_note_upload)

            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            audio = gr.Checkbox(interactive=True, value=False, label="Audio")
            gr.Markdown(Note_markdown)
        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='Video-LLaMA')
            text_input = gr.Textbox(label='User', placeholder='Upload your image/video first, or directly click the examples at the bottom of the page.', interactive=False)
            

    with gr.Column():
        gr.Examples(examples=[
            [f"examples/dog.jpg", "Which breed is this dog? "],
            [f"examples/JonSnow.jpg", "Who's the man on the right? "],
            [f"examples/Statue_of_Liberty.jpg", "Can you tell me about this building? "],
        ], inputs=[image, text_input])

        gr.Examples(examples=[
            [f"examples/skateboarding_dog.mp4", "What is the dog doing? "],
            [f"examples/birthday.mp4", "What is the boy doing? "],
            [f"examples/IronMan.mp4", "Is the guy in the video Iron Man? "],
        ], inputs=[video, text_input])
        
    gr.Markdown(cite_markdown)
    upload_button.click(upload_imgorvideo, [video, image, text_input, chat_state,chatbot,audio], [video, image, text_input, upload_button, chat_state, img_list,chatbot])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, image, text_input, upload_button, chat_state, img_list], queue=False)
    
demo.launch(share=True, enable_queue=False)


# %%
