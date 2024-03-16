import torch
import pickle  # to load all our vocabularies and language decoder/encoder
import sentencepiece as spm
import re
import streamlit as st

EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128


# to merge digits during inference
def merge_digits(text):
    # regular expression pattern to match separated digits
    pattern = r"\b\d+(?:\s+\d+)*(?:\s*:\s*\d+(?:\s+\d+)*)*\b"

    # function to replace the matched separated digits
    def replace(match):
        parts = re.split(r"\s*:\s*", match.group())
        merged_parts = ["".join(part.split()) for part in parts]
        return ":".join(merged_parts)

    # Perform the replacement using the regular expression
    merged_text = re.sub(pattern, replace, text)

    return merged_text


# to split during inference
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


# loading our tokenizer models
sp_ba = spm.SentencePieceProcessor()
sp_ba.Load("load\\target.model")
sp_ru = spm.SentencePieceProcessor()
sp_ru.Load("load\\source.model")


# loading our vocabularies
with open("load\\ru_input_lang.pkl", "rb") as f:
    ru_input_lang = pickle.load(f)
with open("load\\ru_output_lang.pkl", "rb") as f:
    ru_output_lang = pickle.load(f)

with open("load\\ba_input_lang.pkl", "rb") as f:
    ba_input_lang = pickle.load(f)
with open("load\\ba_output_lang.pkl", "rb") as f:
    ba_output_lang = pickle.load(f)

# loading both language encoders and decoders
with open("load\\ru_encoder.pkl", "rb") as f:
    ru_encoder = pickle.load(f)
with open("load\\ru_decoder.pkl", "rb") as f:
    ru_decoder = pickle.load(f)

with open("load\\ba_encoder.pkl", "rb") as f:
    ba_encoder = pickle.load(f)
with open("load\\ba_decoder.pkl", "rb") as f:
    ba_decoder = pickle.load(f)


# loading training parameters
checkpoint_ru = torch.load("load\\checkpoint_90_epoch.pth")
checkpoint_ba = torch.load("load\\checkpoint_90_ba_epoch.pth")

ru_encoder.load_state_dict(checkpoint_ru["encoder_state_dict"])
ru_decoder.load_state_dict(checkpoint_ru["decoder_state_dict"])

ba_encoder.load_state_dict(checkpoint_ba["encoder_state_dict"])
ba_decoder.load_state_dict(checkpoint_ba["decoder_state_dict"])


# setting up evalution mode
ru_encoder.eval()
ru_decoder.eval()

ba_encoder.eval()
ba_decoder.eval()


def translate_sentence(input_sentence, language):
    if language == "ru":
        input_sentence = input_sentence.strip()
        input_sentence = sp_ru.encode_as_pieces(input_sentence)
        input_sentence = " ".join([token for token in input_sentence])
        translated = ""
        output_words, _ = evaluate(
            ru_encoder, ru_decoder, input_sentence, ru_input_lang, ru_output_lang
        )
        for line in output_words:
            line = sp_ba.decode_pieces(line)
            if line in [",", ".", "!"]:
                translated = translated[:-1]
            translated += line
            translated += " "
        translated = translated[:-6]
        translated = merge_digits(translated)
        return translated

    if language == "ba":
        input_sentence = input_sentence.strip()
        input_sentence = sp_ba.encode_as_pieces(input_sentence)
        input_sentence = " ".join([token for token in input_sentence])
        translated = ""
        output_words, _ = evaluate(
            ba_encoder, ba_decoder, input_sentence, ba_input_lang, ba_output_lang
        )
        for line in output_words:
            line = sp_ru.decode_pieces(line)
            if line in [",", ".", "!"]:
                translated = translated[:-1]
            translated += line
            translated += " "
        translated = translated[:-6]
        translated = merge_digits(translated)
        return translated


# setting up Streamlit app
st.title("Simple Translation App")

language = st.radio("Select language pair:", ("Russian-Bashkir", "Bashkir-Russian"))

input_sentence = st.text_input("Enter a sentence to translate:")
if st.button("Translate"):
    if input_sentence:
        if language == "Russian-Bashkir":
            translated_sentence = translate_sentence(input_sentence, "ru")
        else:
            translated_sentence = translate_sentence(input_sentence, "ba")

        st.write("Translated sentence:", translated_sentence)
