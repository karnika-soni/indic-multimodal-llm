"""
Tokenizer wrapper for Indic language models.

Supports:

- SentencePiece tokenizers
- encode()
- decode()
- vocabulary size

Designed for morphologically rich languages
like Kannada.
"""

import sentencepiece as spm



class IndicTokenizer:


    def __init__(
        self,
        model_path
    ):

        self.sp = spm.SentencePieceProcessor()

        self.sp.load(
            model_path
        )


    def encode(
        self,
        text,
        add_bos=True,
        add_eos=True
    ):

        """
        Converts text into token IDs.

        Example:

        ಕರ್ನಾಟಕದಲ್ಲಿ

        becomes:

        [1203,450,23]

        """


        tokens = self.sp.encode(
            text,
            out_type=int
        )


        if add_bos:

            tokens = [
                self.bos_id()
            ] + tokens


        if add_eos:

            tokens.append(
                self.eos_id()
            )


        return tokens



    def decode(
        self,
        ids
    ):

        """
        Converts token IDs back to text.
        """

        return self.sp.decode(
            ids
        )



    def vocab_size(
        self
    ):

        return self.sp.vocab_size()



    def pad_id(
        self
    ):

        return self.sp.pad_id()



    def bos_id(
        self
    ):

        return self.sp.bos_id()



    def eos_id(
        self
    ):

        return self.sp.eos_id()
