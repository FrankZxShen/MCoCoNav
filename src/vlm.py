import time
import logging
import torch
import numpy as np
import io
from PIL import Image
# from prismatic import load


"""
This script is designed to mimic the OpenAI API interface with CogVLM2 Chat
It demonstrates how to integrate image and text-based input to generate a response.
Currently, the model can only handle a single image.
Therefore, do not use this script to process multiple images in one conversation. (includes images from history)
And it only works on the chat model, not the base model.
"""
import requests
import json
import base64

class CogVLM2:
    def __init__(self, base_url):
        self.base_url = base_url

    def create_chat_completion(self, model, messages, temperature=0.8, max_tokens=2048, top_p=0.8, use_stream=False):
        """
        This function sends a request to the chat API to generate a response based on the given messages.

        Args:
            model (str): The name of the model to use for generating the response.
            messages (list): A list of message dictionaries representing the conversation history.
            temperature (float): Controls randomness in response generation. Higher values lead to more random responses.
            max_tokens (int): The maximum length of the generated response.
            top_p (float): Controls diversity of response by filtering less likely options.
            use_stream (bool): Determines whether to use a streaming response or a single response.

        The function constructs a JSON payload with the specified parameters and sends a POST request to the API.
        It then handles the response, either as a stream (for ongoing responses) or a single message.
        """

        data = {
            "model": model,
            "messages": messages,
            "stream": use_stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = requests.post(f"{self.base_url}/v1/chat/completions", json=data, stream=use_stream)
        if response.status_code == 200:
            if use_stream:
                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')[6:]
                        try:
                            response_json = json.loads(decoded_line)
                            content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            scores = decoded_line.get("choices", [{}])[0].get("delta", {}).get("scores", "")
                            # logging.info(f"VLM output: {content}")
                        except:
                            print("Special Token:", decoded_line)
                return scores, content
            else:
                # 处理非流式响应
                decoded_line = response.json()
                content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
                scores = decoded_line.get("choices", [{}])[0].get("message", "").get("scores", "")
                # logging.info(f"decoded line: {decoded_line}")
                # logging.info(f"VLM output: {content}")
                return scores, content
        else:
            print("Error:", response.status_code)
            return None
    
    def encode_image(self, image):
        """
        Encodes an image file into a base64 string.
        Args:
            image_path (str): The path to the image file.

        This function opens the specified image file, reads its content, and encodes it into a base64 string.
        The base64 encoding is used to send images over HTTP as text.

        Change to PIL
        """

        # with open(image_path, "rb") as image_file:
        #     return base64.b64encode(image_file.read()).decode("utf-8")
        img = Image.fromarray(image)
        img_byte_arr = io.BytesIO()

        img.save(img_byte_arr, format='PNG')

        # 将字节流的内容转换为 base64 编码的字符串
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        return base64_image
    
    def simple_image_chat(self, User_Prompt, return_string_probabilities=None, use_stream=False, img=None):
        """
        Facilitates a simple chat interaction involving an image.
        COT1

        Args:
            use_stream (bool): Specifies whether to use streaming for chat responses.
            img_path (str): Path to the image file to be included in the chat.

        This function encodes the specified image and constructs a predefined conversation involving the image.
        It then calls `create_chat_completion` to generate a response from the model.
        The conversation includes asking about the content of the image and a follow-up question.
        """

        img_url = f"data:image/jpeg;base64,{self.encode_image(img)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": User_Prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        },
                    },
                ],
                "return_string_probabilities": return_string_probabilities,
            },
            # {
            #     "role": "assistant",
            #     "content": "The image displays a wooden boardwalk extending through a vibrant green grassy wetland. The sky is partly cloudy with soft, wispy clouds, indicating nice weather. Vegetation is seen on either side of the boardwalk, and trees are present in the background, suggesting that this area might be a natural reserve or park designed for ecological preservation and outdoor recreation. The boardwalk allows visitors to explore the area without disturbing the natural habitat.",
            # },
            # {
            #     "role": "user",
            #     "content": "Do you think this is a spring or winter photo?"
            # },
        ]
        scores, pred = self.create_chat_completion("cogvlm2", messages=messages, use_stream=use_stream)

        return scores, pred

    def CoT2(self, User_Prompt1, User_Prompt2, cot_pred1, return_string_probabilities=None, use_stream=False, img=None):
        """
        Facilitates a simple chat interaction involving an image.

        Args:
            use_stream (bool): Specifies whether to use streaming for chat responses.
            img_path (str): Path to the image file to be included in the chat.

        This function encodes the specified image and constructs a predefined conversation involving the image.
        It then calls `create_chat_completion` to generate a response from the model.
        The conversation includes asking about the content of the image and a follow-up question.
        """

        img_url = f"data:image/jpeg;base64,{self.encode_image(img)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": User_Prompt1,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        },
                    },
                ],
                "return_string_probabilities": return_string_probabilities,
            },
            {
                "role": "assistant",
                "content": cot_pred1,
            },
            {
                "role": "user",
                "content": User_Prompt2,
                "return_string_probabilities": return_string_probabilities,
            },
        ]
        scores, pred = self.create_chat_completion("cogvlm2", messages=messages, use_stream=use_stream)

        return scores, pred
    
    def CoT3(self, User_Prompt1, User_Prompt2, User_Prompt3, cot_pred1, cot_pred2, return_string_probabilities=None, use_stream=False, img=None):
        """
        Facilitates a simple chat interaction involving an image.

        Args:
            use_stream (bool): Specifies whether to use streaming for chat responses.
            img_path (str): Path to the image file to be included in the chat.

        This function encodes the specified image and constructs a predefined conversation involving the image.
        It then calls `create_chat_completion` to generate a response from the model.
        The conversation includes asking about the content of the image and a follow-up question.
        """

        img_url = f"data:image/jpeg;base64,{self.encode_image(img)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": User_Prompt1,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        },
                    },
                ],
                "return_string_probabilities": None,
            },
            {
                "role": "assistant",
                "content": cot_pred1,
            },
            {
                "role": "user",
                "content": User_Prompt2,
                "return_string_probabilities": None,
            },
            {
                "role": "assistant",
                "content": cot_pred2,
            },
            {
                "role": "user",
                "content": User_Prompt3,
                "return_string_probabilities": return_string_probabilities,
            },
        ]
        scores, pred = self.create_chat_completion("cogvlm2", messages=messages, use_stream=use_stream)

        return scores, pred
    
    def ToT3(self, User_Prompt1, User_Prompt2, User_Prompt3, cot_pred1, cot_pred2, return_string_probabilities=None, use_stream=False, img=None):
        """
        Facilitates a simple chat interaction involving an image.

        Args:
            use_stream (bool): Specifies whether to use streaming for chat responses.
            img_path (str): Path to the image file to be included in the chat.

        This function encodes the specified image and constructs a predefined conversation involving the image.
        It then calls `create_chat_completion` to generate a response from the model.
        The conversation includes asking about the content of the image and a follow-up question.
        """

        img_url = f"data:image/jpeg;base64,{self.encode_image(img)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": User_Prompt1,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        },
                    },
                ],
                "return_string_probabilities": return_string_probabilities,
            },
            {
                "role": "assistant",
                "content": cot_pred1,
            },
            {
                "role": "user",
                "content": User_Prompt2,
                "return_string_probabilities": return_string_probabilities,
            },
            {
                "role": "assistant",
                "content": cot_pred2,
            },
            {
                "role": "user",
                "content": User_Prompt3,
                "return_string_probabilities": return_string_probabilities,
            },
        ]
        scores, pred = self.create_chat_completion("cogvlm2", messages=messages, use_stream=use_stream)

        return scores, pred