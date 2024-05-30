import logging

import tiktoken
from transformers import AutoTokenizer

import gradio as gr

logger = logging.getLogger(__name__)  # noqa


def load_test_phrases(filename):
    with open(f"./data/{filename}", "r", encoding="utf-8") as file:
        return file.read().splitlines()


models = ["meta-llama/Llama-2-7b-chat-hf",           # LLAMA-2
          "beomi/llama-2-ko-7b",                     # LLAMA-2-ko
          "ai4bharat/Airavata",                      # ARIVATA
          "openaccess-ai-collective/tiny-mistral",   # Mistral
          "gpt-3.5-turbo",                           # GPT3.5
          "meta-llama/Meta-Llama-3-8B-Instruct",     # LLAMA-3
          "CohereForAI/aya-23-8B",                   # AYA
          "google/gemma-1.1-2b-it",                  # GEMMA
          "gpt-4o",                                  # GPT4o
          "TWO/sutra-mlt256-v2"]                     # SUTRA

test_phrase_set = [
    "I am going for a walk later today",
    "நாங்கள் சந்திரனுக்கு ராக்கெட் பயணத்தில் இருக்கிறோம்",

    "중성자 산란을 다섯 문장으로 설명해주세요",  # Korean,

    "मुझे पाँच वाक्यों में न्यूट्रॉन प्रकीर्णन की व्याख्या दीजिए",  # Hindi
    "mujhe paanch vaakyon mein nyootron prakeernan kee vyaakhya deejie",

    "আমাকে পাঁচটি বাক্যে নিউট্রন বিচ্ছুরণের একটি ব্যাখ্যা দিন",  # Bengali/Bangla
    "Amake pamcati bakye ni'utrana bicchuranera ekati byakhya dina",

    "મને પાંચ વાક્યોમાં ન્યુટ્રોન સ્કેટરિંગની સમજૂતી આપો",  # Gujarati
    "Mane panca vakyomam n'yutrona sketaringani samajuti apo",

    "நியூட்ரான் சிதறல் பற்றிய விளக்கத்தை ஐந்து வாக்கியங்களில் கொடுங்கள்",  # Tamil
    "Niyutran citaral parriya vilakkattai aintu vakkiyankalil kotunkal",

    "मला पाच वाक्यात न्यूट्रॉन स्कॅटरिंगचे स्पष्टीकरण द्या",  # Marathi

    "ఐదు వాక్యాలలో న్యూట్రాన్ స్కాటరింగ్ గురించి నాకు వివరణ ఇవ్వండి",  # Telugu
]

test_phrase_set_long_1 = load_test_phrases('multilingualphrases01.txt')
test_phrase_set_long_2 = load_test_phrases('multilingualphrases02.txt')
test_phrase_set_long_3 = load_test_phrases('multilingualphrases03.txt')


def generate_tokens_as_table(text):
    table = []
    for model in models:
        if 'gpt' not in model:
            tokenizer = AutoTokenizer.from_pretrained(model)
            tokens = tokenizer.encode(text, add_special_tokens=False)
        else:
            tokenizer = tiktoken.encoding_for_model(model)
            tokens = tokenizer.encode(text)
        decoded = [tokenizer.decode([t]) for t in tokens]
        table.append([model] + decoded)
    return table


def generate_tokenizer_table(text):
    if not text:
        return []

    token_counts = {model: 0 for model in models}
    vocab_size = {model: 0 for model in models}

    for model in models:
        if 'gpt' not in model:
            tokenizer = AutoTokenizer.from_pretrained(model)
            vocab_size[model] = tokenizer.vocab_size
        else:
            tokenizer = tiktoken.encoding_for_model(model)
            vocab_size[model] = tokenizer.n_vocab

        token_counts[model] += len(tokenizer.encode(text))

    word_count = len(text.split(' '))

    output = []
    for m in models:
        row = [m, vocab_size[m], word_count, token_counts[m], f"{token_counts[m] / word_count:0.2f}"]
        output.append(row)

    return output


def generate_split_token_table(text):
    if not text:
        return gr.Dataframe()

    table = generate_tokenizer_table(text)
    return gr.Dataframe(
        table,
        headers=['tokenizer', 'v size', '#word', '#token', '#tokens/word'],
        datatype=["str", "number", "str"],
        row_count=len(models),
        col_count=(5, "fixed"),
    )


with gr.Blocks() as sutra_token_count:
    gr.Markdown(
        """
        # SUTRA Multilingual Tokenizer Specs & Stats.
        ## Tokenize paragraphs in multiple languages and compare token counts.
        """)
    textbox = gr.Textbox(label="Input Text")
    submit_button = gr.Button("Submit")
    output = gr.Dataframe()
    examples = [
        [' '.join(test_phrase_set_long_1)],
        [' '.join(test_phrase_set_long_2)],
        [' '.join(test_phrase_set_long_3)],
    ]
    gr.Examples(examples=examples, inputs=[textbox])
    submit_button.click(generate_split_token_table, inputs=[textbox], outputs=[output])


def generate_tokens_table(text):
    table = generate_tokens_as_table(text)
    cols = len(table[0])
    return gr.Dataframe(
        table,
        headers=['model'] + [str(i) for i in range(cols - 1)],
        row_count=2,
        col_count=(cols, "fixed"),
    )


with gr.Blocks() as sutra_tokenize:
    gr.Markdown(
        """
        # SUTRA Multilingual Tokenizer Sentence Inspector.
        ## Tokenize a sentence with various tokenizers and inspect how it's broken down.
        """)
    textbox = gr.Textbox(label="Input Text")
    submit_button = gr.Button("Submit")
    output = gr.Dataframe()
    examples = test_phrase_set
    gr.Examples(examples=examples, inputs=[textbox])
    submit_button.click(generate_tokens_table, inputs=[textbox], outputs=[output])


if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(
                """
                ## <img src="https://playground.two.ai/sutra.svg" height="20"/>
                """
            )
        with gr.Row():
            gr.TabbedInterface(
                interface_list=[sutra_tokenize, sutra_token_count],
                tab_names=["Tokenize Text", "Tokenize Paragraphs"]
            )

demo.queue(default_concurrency_limit=5).launch(
    server_name="0.0.0.0",
    allowed_paths=["/"],
)
