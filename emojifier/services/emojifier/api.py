from tensorflow.keras.models import model_from_json
import numpy as np
import pandas as pd
import emoji

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                   }


def embedding_output(X):
    maxLen = 10
    emb_dim = 50
    embedding_out = np.zeros((X.shape[0],maxLen,emb_dim))
    
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for ij in range(len(X[ix])):
            try:
                embedding_out[ix][ij] = embeddings_index[X[ix][ij].lower()]
            except:
                embedding_out[ix][ij] = np.zeros((emb_dim,))
    return embedding_out

embeddings_index={}
with open("services/emojifier/glove.6B.50d.txt",encoding='utf-8') as f:
    for line in f:
        value = line.split(" ")
        word = value[0]
        coef = np.asarray(value[1:],dtype = 'float32')
        embeddings_index[word] = list(coef)


with open("services/emojifier/best_emoji_model.json","r") as f:
    model = model_from_json(f.read())
model.load_weights("services/emojifier/best_emoji_model.h5")
model._make_predict_function()


def predict(test_str):
    X = pd.Series([test_str])
    emb_X = embedding_output(X)
    p = model.predict_classes(emb_X)
    return emoji.emojize(emoji_dictionary[str(p[0])])

if __name__=="__main__":
    print(predict("Work is Hard"))