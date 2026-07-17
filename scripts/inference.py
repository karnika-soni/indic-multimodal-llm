import torch

from inference.generate import generate_text

from utils.device import get_device



def main():

    device=get_device()


    model=torch.load(
        "checkpoints/model.pt"
    )


    prompt="ಬೆಂಗಳೂರು"


    tokens = tokenizer.encode(
        prompt
    )


    idx=torch.tensor(
        [tokens]
    ).to(device)



    output = generate_text(
        model,
        idx,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    )


    print(
        tokenizer.decode(
            output[0].tolist()
        )
    )



if __name__=="__main__":

    main()
