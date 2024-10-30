from fastapi import FastAPI
from pydantic import BaseModel
import re
import numpy as np
from collections import Counter
from nltk.util import ngrams
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import string
import torch

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Đường dẫn tới mô hình trên Hugging Face
model1_path = "Trinity2105/modelkerasVNcorect"
model2_path = "Trinity2105/finalprojectVNcorect"

# Tải mô hình Keras từ Hugging Face
model = TFAutoModelForSequenceClassification.from_pretrained(model1_path)
tokenizer1 = AutoTokenizer.from_pretrained(model1_path)

# Cấu hình Transformers sử dụng GPU nếu có
device = 0 if torch.cuda.is_available() else -1
corrector = pipeline("text2text-generation", model=model2_path, tokenizer=model2_path, device=device)

# Các biến
NGRAM = 2
MAXLEN = 40
alphabet = ['\x00', ' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'đ', 'Á', 'À', 'Ả', 'Ã', 'Ạ', 'Â', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ă', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ', 'Ô', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ơ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ', 'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ê', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ư', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị', 'Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ', 'Đ']
accepted_char = list(string.digits + ''.join(alphabet))

# Các hàm trợ giúp
def encoder_data(text, maxlen=MAXLEN):
    x = np.zeros((maxlen, len(alphabet)))
    for i, c in enumerate(text[:maxlen]):
        if c in alphabet:
            x[i, alphabet.index(c)] = 1
    if i < maxlen - 1:
        for j in range(i + 1, maxlen):
            x[j, 0] = 1
    return x

def decoder_data(x):
    x = x.argmax(axis=-1)
    return ''.join(alphabet[i] for i in x)

def batch_predict(ngrams_batch):
    batch_input = np.array([encoder_data(' '.join(ngram)) for ngram in ngrams_batch])
    preds = model.predict(batch_input, verbose=0)
    return [decoder_data(pred).strip('\x00') for pred in preds]

# Hàm chính để sửa lỗi chính tả
def correct_sentence(sentence):
    def separate_words(text):
        return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)

    def nltk_ngrams(words, n=2):
        return ngrams(words.split(), n)

    sentence = separate_words(sentence)
    for i in sentence:
        if i not in accepted_char:
            sentence = sentence.replace(i, " ")

    ngrams_list = list(nltk_ngrams(sentence, n=NGRAM))
    guessed_ngrams = batch_predict(ngrams_list)

    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])

    if not candidates or all(len(c) == 0 for c in candidates):
        return "Không có từ nào để sửa."

    first_guess = ' '.join(c.most_common(1)[0][0] for c in candidates if c)

    # Chạy mô hình transformers
    predictions = corrector(first_guess, max_length=512)
    final_output = predictions[0]['generated_text']
    return final_output

# Định nghĩa cấu trúc request và response
class SentenceRequest(BaseModel):
    sentence: str

class SentenceResponse(BaseModel):
    corrected_sentence: str

# Định nghĩa route cho API
@app.post("/correct_sentence", response_model=SentenceResponse)
async def api_correct_sentence(request: SentenceRequest):
    corrected = correct_sentence(request.sentence)
    return SentenceResponse(corrected_sentence=corrected)
