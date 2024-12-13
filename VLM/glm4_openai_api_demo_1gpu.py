import gc
import time
import base64
import random

from contextlib import asynccontextmanager
from typing import List, Literal, Union, Tuple, Optional
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from PIL import Image
from io import BytesIO
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

MODEL_PATH = 'THUDM/glm-4v-9b'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends, which is essential for efficient resource management in GPU environments.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    """
    A Pydantic model representing a model card, which provides metadata about a machine learning model.
    It includes fields like model ID, owner, and creation time.
    """
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ImageUrl(BaseModel):
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = Union[TextContent, ImageUrlContent]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None
    return_string_probabilities: Optional[str] = None


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None
    scores: Optional[List[float]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    scores: Optional[List[float]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    An endpoint to list available models. It returns a list of model cards.
    This is useful for clients to query and understand what models are available for use.
    """
    model_card = ModelCard(id="cogvlm2-19b")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
    )

    if request.messages[-1].return_string_probabilities == "[Yes, No]":
        gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        return_string_probabilities=['Yes', 'No'],
        )
    elif request.messages[-1].return_string_probabilities == "[A, B, C, D]":
        gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        return_string_probabilities=['A', 'B', 'C', 'D'],
        )
    elif request.messages[-1].return_string_probabilities == "[a, b, c, d]":
        gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        return_string_probabilities=[chr(ord('a') + i) for i in range(26)],
        )
    elif not request.messages[-1].return_string_probabilities:
        gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        return_string_probabilities=None,
        )


    if request.stream:
        generate = predict(request.model, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")
    response = generate_cogvlm(model, tokenizer, gen_params)

    usage = UsageInfo()

    message = ChatMessageResponse(
        role="assistant",
        content=response["text"],
        scores=response["softmax_scores"],
    )
    logger.debug(f"==== message ====\n{message}")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)


async def predict(model_id: str, params: dict):
    """
    Handle streaming predictions. It continuously generates responses for a given input stream.
    This is particularly useful for real-time, continuous interactions with the model.
    """

    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_cogvlm(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode
        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
            scores=new_response["softmax_scores"]
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))


def generate_cogvlm(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, params: dict):
    """
    Generates a response using the CogVLM2 model. It processes the chat history and image data, if any,
    and then invokes the model to generate a response.
    """

    response = None

    for response in generate_stream_cogvlm(model, tokenizer, params):
        pass
    return response


def process_history_and_images(messages: List[ChatMessageInput]) -> Tuple[
    Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
    """
    Process history messages to extract text, identify the last user query,
    and convert base64 encoded image URLs to PIL images.

    Args:
        messages(List[ChatMessageInput]): List of ChatMessageInput objects.
    return: A tuple of three elements:
             - The last user query as a string.
             - Text history formatted as a list of tuples for the model.
             - List of PIL Image objects extracted from the messages.
    """

    formatted_history = []
    image_list = []
    last_user_query = ''

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = ' '.join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if image_url.startswith("data:image/jpeg;base64,"):
                        base64_encoded_image = image_url.split("data:image/jpeg;base64,")[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                        image_list.append(image)

        if role == 'user':
            if i == len(messages) - 1:  # 最后一条用户消息
                last_user_query = text_content
            else:
                formatted_history.append((text_content, ''))
        elif role == 'assistant':
            if formatted_history:
                if formatted_history[-1][1] != '':
                    assert False, f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, f"assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history, image_list

def process_history_and_images_glm4(messages: List[ChatMessageInput]) -> Tuple[
    Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:

    glm4_messages = []
    # image_list = []
    
    for i, message in enumerate(messages):
        role = message.role
        content = message.content
        if isinstance(content, list):  # text
            text_content = ' '.join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content
        
        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if image_url.startswith("data:image/jpeg;base64,"):
                        base64_encoded_image = image_url.split("data:image/jpeg;base64,")[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                        image_list=image
        if i==0:        
            messages_final = {'role': role, 'content': text_content, 'image': image_list}
        else:
            messages_final = {'role': role, 'content': text_content}
        glm4_messages.append(messages_final)

    return glm4_messages


@torch.inference_mode()
def generate_stream_cogvlm(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, params: dict):
    """
    Generates a stream of responses using the CogVLM model in inference mode.
    It's optimized to handle continuous input-output interactions with the model in a streaming manner.
    """
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    query, history, image_list = process_history_and_images(messages)

    logger.debug(f"==== request ====\n{query}")

    messages = process_history_and_images_glm4(messages)

    model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(next(model.parameters()).device)


    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60.0,
        skip_prompt=True,
        skip_special_tokens=True
    )
    gen_kwargs = {
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p if temperature > 1e-5 else 0,
        'streamer': streamer,
        'output_scores': True,
        'return_dict_in_generate': True,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    generated_text = ""

    # === Generation Utilities ===
    #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
    string2idx = {}
    space_ = [" A", " B", " C", " D"]
    ALL_space_ = [" A", " B", " C", " D", " E", " F", " G", " H", " I", " J", " K", " L", " M", " N", " O", " P", " Q", " R", " S", " T", " U", " V", " W", " X", " Y", " Z"]
    ALL_nums_ = [" a", " b", " c", " d", " e", " f", " g", " h", " i", " j", " k", " l", " m", " n", " o", " p", " q", " r", " s", " t", " u", " v", " w", " x", " y", " z"]
    # line_ = ["\nA", "\nB", "\nC", "\nD", "\nE", "\nF", "\nG", "\nH", "\nI", "\nJ", "\nK", "\nL", "\nM", "\nN", "\nO", "\nP", "\nQ", "\nR", "\nS", "\nT", "\nU", "\nV", "\nW", "\nX", "\nY", "\nZ"]
    # space_line_ = [' \nA', ' \nB', ' \nC', ' \nD', ' \nE', ' \nF', ' \nG', ' \nH', ' \nI', ' \nJ', ' \nK', ' \nL', ' \nM', ' \nN', ' \nO', ' \nP', ' \nQ', ' \nR', ' \nS', ' \nT', ' \nU', ' \nV', ' \nW', ' \nX', ' \nY', ' \nZ']
    # nums = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    # nums_space_ = [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', ' 10', ' 11', ' 12', ' 13', ' 14', ' 15', ' 16', ' 17', ' 18', ' 19', ' 20', ' 21', ' 22', ' 23', ' 24', ' 25', ' 26']
    # nums_line_ = ['\n1', '\n2', '\n3', '\n4', '\n5', '\n6', '\n7', '\n8', '\n9', '\n10', '\n11', '\n12', '\n13', '\n14', '\n15', '\n16', '\n17', '\n18', '\n19', '\n20', '\n21', '\n22', '\n23', '\n24', '\n25', '\n26']
    # nums_space_line_ = [' \n1', ' \n2', ' \n3', ' \n4', ' \n5', ' \n6', ' \n7', ' \n8', ' \n9', ' \n10', ' \n11', ' \n12', ' \n13', ' \n14', ' \n15', ' \n16', ' \n17', ' \n18', ' \n19', ' \n20', ' \n21', ' \n22', ' \n23', ' \n24', ' \n25', ' \n26']
    ALL_PARAMAS = ["True", "False", "Yes", "No", " Yes", " No", ] + \
        [chr(ord("A") + i) for i in range(26)] + space_ + ALL_space_  \
        + [chr(ord("a") + i) for i in range(26)] + ALL_nums_
    for trigger_string in ALL_PARAMAS:
        token_idx_list = tokenizer.encode(trigger_string, add_special_tokens=False)
        assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
        string2idx[trigger_string] = token_idx_list[0]
    
    if not params['return_string_probabilities']:
        total_len = 0
        generated_text = ""
        with torch.no_grad():
            model.generate(**model_inputs, **gen_kwargs)
            for next_text in streamer:
                generated_text += next_text
                yield {
                    "text": generated_text,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": total_len - 0,
                        "total_tokens": total_len,
                    },
                    "softmax_scores": [0, 0]
                }
        ret = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": total_len - 0,
                "total_tokens": total_len,
            },
            "softmax_scores": [0, 0]
        }
        yield ret
    
    else:
        with torch.no_grad():
            full_out_dict = model.generate(**model_inputs, **gen_kwargs)
            token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0) 
            torch.set_printoptions(profile="full")
            torch.set_printoptions(profile="default") # reset
            slice_idxs = torch.tensor([string2idx[s] for s in params['return_string_probabilities']])
            string_probs_unnormalized = token_probs[slice_idxs]
            if string_probs_unnormalized.sum() == 0.0 and len(slice_idxs) == 2:
                slice_idxs = torch.tensor([string2idx[s] for s in [" Yes", " No"]])
                string_probs_unnormalized = token_probs[slice_idxs]
            elif string_probs_unnormalized.sum() == 0.0 and len(slice_idxs) == 4:
                slice_idxs = torch.tensor([string2idx[s] for s in [" A", " B", " C", " D"]])
                string_probs_unnormalized = token_probs[slice_idxs]
            elif string_probs_unnormalized.sum() == 0.0 and len(slice_idxs) > 4:
                slice_idxs = torch.tensor([string2idx[s] for s in ALL_nums_])
                string_probs_unnormalized = token_probs[slice_idxs]
            

            if string_probs_unnormalized.sum() == 0.0:
                rand_list = [random.random() for _ in range(len(slice_idxs))]
                total_sum = sum(rand_list)
                gen_probabilities = [num/total_sum for num in rand_list]
            else:
                string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                gen_probabilities = string_probs.to(torch.float).cpu().numpy().tolist()

            # model.generate(**inputs, **gen_kwargs)
            for next_text in streamer:
                generated_text += next_text

                yield {
                    "text": generated_text,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": total_len - 0,
                        "total_tokens": total_len,
                    },
                    "softmax_scores": gen_probabilities
                }
        ret = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": total_len - 0,
                "total_tokens": total_len,
            },
            "softmax_scores": gen_probabilities
        }
        yield ret


gc.collect()
torch.cuda.empty_cache()
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_PATH,
    #     trust_remote_code=True,
    #     torch_dtype=torch_type,
    # ).to(DEVICE).eval()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        low_cpu_mem_usage=True
    ).eval()
    # device_map = infer_auto_device_map(
    #     model=model,
    #     max_memory={i: "10GiB" for i in range(torch.cuda.device_count())},
    #     # set 23GiB for each GPU, depends on your GPU memory, you can adjust this value
    #     no_split_module_classes=["CogVLMDecoderLayer"]
    # )
    # model = load_checkpoint_and_dispatch(model, MODEL_PATH, device_map=device_map, dtype=torch_type)
    # model = model.eval()

    uvicorn.run(app, host='127.0.0.1', port=31511, workers=1)
