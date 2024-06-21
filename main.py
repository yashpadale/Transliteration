from transformer import *
from keras.models import load_model
import pickle
maxlen=60
f=r'C:\Users\Yash\PycharmProjects\pythonProject7\output_english.txt',



model, dictionary, maxlen = transformer(maxlen=maxlen,
                                                 embed_dim=256,
                                                 num_heads=32,
                                                 ff_dim=256,
                                                 num_blocks=5,
                                                 dropout_rate=0.1,
                                                 input_file="t.txt",
                                                 per=0.85,
                                                 batch_size=64,
                                                 epochs=3,
                                                 num_decoders=1,num_encoders=1)
model.save('transformer_model.h5')

with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)





custom_objects = {
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'TransformerBlock': TransformerBlock,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding
}
loaded_model = load_model('transformer_model.h5', custom_objects=custom_objects)
with open('dictionary.pkl', 'rb') as f:
    loaded_dictionary = pickle.load(f)


def generate_text(s1):
    s1='<start> '+s1+' <end> '
    s1=pad_segments(content=s1,maxlen=maxlen)
    s1_=s1.split(' ')
    #print('prompt - ')
    #print(s1)
    words = query_gen_sentences(query=s1,
                                model=loaded_model, dictionary=loaded_dictionary, maxlen=maxlen)


    for i in range(len(s1_)):
        w1 = query_gen_sentences(query=words[-1],
                                 model=loaded_model, dictionary=loaded_dictionary, maxlen=maxlen)
        words.append(w1[0])



    return words


def pr(output):
    respone=''
    for i in range(len(output)):
        k=output[i]
        p=k.split(' ')
        respone+=' '+p[-1]+' '
    return respone

while True:
    i=input("Enter : ")
    o = generate_text(s1=i)
    #print(o)
    sentence=pr(o)
    print(sentence)




