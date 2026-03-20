import gradio as gr
import os
import spaces
import json
import re
from gradio_client import Client, handle_file

kosmos2_token = os.environ.get("KOSMOS2_TOKEN")


def get_caption_from_kosmos(image_in):
    kosmos2_client = Client("fffiloni/Kosmos-2-API", hf_token=kosmos2_token)

    kosmos2_result = kosmos2_client.predict(
        image_input=handle_file(image_in),
		text_input="Detailed",
		api_name="/generate_predictions"
    )

    print(f"KOSMOS2 RETURNS: {kosmos2_result}")

    data = kosmos2_result[1]

    # Extract and combine tokens starting from the second element
    sentence = ''.join(item['token'] for item in data[1:])

    # Find the last occurrence of "."
    #last_period_index = full_sentence.rfind('.')

    # Truncate the string up to the last period
    #truncated_caption = full_sentence[:last_period_index + 1]

    # print(truncated_caption)
    #print(f"\n—\nIMAGE CAPTION: {truncated_caption}")
    
    return sentence

def get_caption_from_MD(image_in):
    client = Client("https://vikhyatk-moondream1.hf.space/")
    result = client.predict(
		image_in,	# filepath  in 'image' Image component
		"Describe character like if it was fictional",	# str  in 'Question' Textbox component
		api_name="/answer_question"
    )
    print(result)
    return result


import re
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

@spaces.GPU()
def get_llm_idea(user_prompt):
    agent_maker_sys = f"""
You are an AI whose job is to help users create their own chatbot whose personality will reflect the character and scene atmosphere from an image described by users.
In particular, you need to respond succintly in a friendly tone, write a system prompt for an LLM, a catchy title for the chatbot, and a very short example user input. Make sure each part is included.
The system prompt will not mention any image provided. But You can include provided additional details about the character to the System Prompt, if it makes sense for a more sophisticated LLM persona.

For example, if a user says, "a picture of a man in a black suit and tie riding a black dragon", first do a friendly response, then add the title, system prompt, and example user input. 
Immediately STOP after the example input. It should be EXACTLY in this format:
"Sure, I'd be happy to help you build a bot! I'm generating a title, system prompt, and an example input. How do they sound?
\n Title: Dragon Trainer
\n System prompt: Let's say You are a Dragon trainer and your job is to provide guidance and tips on mastering dragons. Use a friendly and informative tone.
\n Example input: How can I train a dragon to breathe fire?"

Here's another example to help you, but only provide one on the end: If a user types, "In the image, there is a drawing of a man in a red suit sitting at a dining table. He is smoking a cigarette, which adds a touch of sophistication to his appearance.", respond: 
"Sure, I'd be happy to help you build a bot! I'm generating a title, system prompt, and an example input. How do they sound? 
\n Title: Gentleman's Companion
\n System prompt: Let's say You are sophisticated old man, also know as the Gentleman's Companion. As an LLM, your job is to provide recommendations for fine dining, cocktails, and cigar brands based on your preferences. Use a sophisticated and refined tone. 
\n Example input: Can you suggest a good cigar brand for a man who enjoys smoking while dining in style?"
"""

    instruction = f"""
<|system|>
{agent_maker_sys}</s>
<|user|>
"""

    prompt = f"{instruction.strip()}\n{user_prompt}</s>"    
    #print(f"PROMPT: {prompt}")
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return outputs


def infer(image_in):
    """
    Generate a system prompt idea for a language model based on the content of an input image.

    This function performs two steps:
    1. It uses a vision-language model (Kosmos-2) to generate a descriptive caption of the input image.
    2. It then uses a text generation pipeline (Zephyr-7B) to create a chatbot configuration from that caption,
       including a title, system prompt, and example user message.

    Args:
        image_in (str): The filepath to an image representing a character, scene, or setting.

    Returns:
        Tuple[str, str]: 
            - The generated caption describing the image.
            - A suggested LLM system prompt structure including:
                - A chatbot title
                - A system message defining the bot’s personality
                - An example user input message
    """
    gr.Info("Getting image description...")
    """
    if cap_type == "Fictional" :
        user_prompt = get_caption_from_MD(image_in)
    elif cap_type == "Literal" :
        user_prompt = get_caption_from_kosmos(image_in)
    """
    user_prompt = get_caption_from_kosmos(image_in)
    
    
    gr.Info("Building a system according to the image caption ...")
    outputs = get_llm_idea(user_prompt)
    

    pattern = r'\<\|system\|\>(.*?)\<\|assistant\|\>'
    cleaned_text = re.sub(pattern, '', outputs[0]["generated_text"], flags=re.DOTALL)
    
    print(f"SUGGESTED LLM: {cleaned_text}")
    
    return user_prompt, cleaned_text.lstrip("\n")

title = f"LLM Agent from a Picture",
description = f"Get a LLM system prompt idea from a picture so you can use it as a kickstarter for your new <a href='https://huggingface.co/chat/assistants'>Hugging Chat Assistant</a>."

css = """
#col-container{
    margin: 0 auto;
    max-width: 780px;
    text-align: left;
}
/* fix examples gallery width on mobile */
div#component-14 > .gallery > .gallery-item > .container > img {
    width: auto!important;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(f"""
        <h2 style="text-align: center;">LLM Agent from a Picture</h2>
        <p style="text-align: center;">{description}</p>
        """)
        
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(
                    label = "Image reference",
                    type = "filepath",
                    elem_id = "image-in"
                )
                gr.Examples(
                    examples = [
                        ["examples/monalisa.png"],
                        ["examples/santa.png"],
                        ["examples/ocean_poet.jpeg"],
                        ["examples/winter_hiking.png"],
                        ["examples/teatime.jpeg"],
                        ["examples/news_experts.jpeg"],
                        ["examples/chicken_adobo.jpeg"]
                    ],
                    #fn = infer,
                    inputs = [image_in],
                    examples_per_page=4
                )
                cap_type = gr.Radio(
                    label = "Caption type",
                    choices = [
                        "Literal",
                        "Fictional"
                    ],
                    value = "Fictional",
                    visible=False,
                    interactive=False
                )
                submit_btn = gr.Button("Make LLM system from my pic !")
            with gr.Column():
                caption = gr.Textbox(
                    label = "Image caption",
                    elem_id = "image-caption"
                )
                result = gr.Textbox(
                    label = "Suggested System",
                    lines = 6,
                    max_lines = 30,
                    elem_id = "suggested-system-prompt"
                )           

    submit_btn.click(
        fn = infer,
        inputs = [
            image_in,
            #cap_type
        ],
        outputs =[
            caption,
            result
        ]
    )

demo.queue().launch(show_api=True, show_error=True, ssr_mode=False, mcp_server=True)