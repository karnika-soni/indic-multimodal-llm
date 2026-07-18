"""
Tokenizer comparison experiment.

Evaluates:

- BPE
- WordPiece
- SentencePiece


Metrics:

1. Average tokens per sentence
2. Characters per token
3. Fragmentation score

"""


from tokenizers import Tokenizer

from tokenizer.sentencepiece_tokenizer import (
    SentencePieceTokenizer
)




def evaluate(
    texts,
    encode_function
):


    total_tokens = 0

    total_chars = 0



    for text in texts:


        tokens = encode_function(
            text
        )


        total_tokens += len(tokens)

        total_chars += len(text)



    return {

        "tokens_per_sentence":
        total_tokens / len(texts),


        "chars_per_token":
        total_chars / total_tokens

    }



def main():


    samples = [

        "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ",

        "ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ",

        "தமிழ் ஒரு பழமையான மொழி"

    ]



    sp = SentencePieceTokenizer(

        "tokenizer/indic.model"

    )



    bpe = Tokenizer.from_file(

        "tokenizer/bpe_tokenizer.json"

    )



    wp = Tokenizer.from_file(

        "tokenizer/wordpiece_tokenizer.json"

    )



    results = {}



    results["SentencePiece"] = evaluate(

        samples,

        lambda x:
        sp.encode(x)

    )


    results["BPE"] = evaluate(

        samples,

        lambda x:
        bpe.encode(x).ids

    )


    results["WordPiece"] = evaluate(

        samples,

        lambda x:
        wp.encode(x).ids

    )



    for name,value in results.items():

        print(
            name,
            value
        )



if __name__ == "__main__":

    main()
