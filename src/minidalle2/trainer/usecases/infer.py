from dalle2_pytorch import DALLE2


def infer(dalle2: DALLE2, input_text: str):

    images = dalle2(
        [input_text],
        cond_scale=2.0,  # classifier free guidance strength (> 1 would strengthen the condition)
    )

    return images.to("cpu")[0].permute(1, 2, 0)
