from dalle2_pytorch import (
    CLIP,
    DALLE2,
    Decoder,
    DiffusionPrior,
    DiffusionPriorNetwork,
    Unet,
)

from minidalle2.values.config import Config


def build_clip(config: Config) -> CLIP:
    return CLIP(
        dim_text=512,
        dim_image=512,
        dim_latent=512,
        num_text_tokens=49408,
        text_enc_depth=6,
        text_seq_len=256,
        text_heads=8,
        visual_enc_depth=6,
        visual_image_size=256,
        visual_patch_size=32,
        visual_heads=8,
    ).to(config.device)


def build_prior(config: Config, prior_network=None, clip=None) -> DiffusionPrior:

    if not clip:
        clip = build_clip(config)

    if not prior_network:
        prior_network = DiffusionPriorNetwork(dim=512, depth=6, dim_head=64, heads=8).to(
            config.device
        )

    return DiffusionPrior(net=prior_network, clip=clip, timesteps=100, cond_drop_prob=0.2).to(
        config.device
    )


def build_decoder(config: Config, clip=None) -> DALLE2:
    if not clip:
        clip = build_clip(config)

    unet1 = Unet(dim=128, image_embed_dim=512, cond_dim=128, channels=3, dim_mults=(1, 2, 4, 8)).to(
        config.device
    )

    unet2 = Unet(
        dim=16, image_embed_dim=512, cond_dim=128, channels=3, dim_mults=(1, 2, 4, 8, 16)
    ).to(config.device)

    return Decoder(
        unet=(unet1, unet2),
        image_sizes=(128, 256),
        clip=clip,
        timesteps=100,
        image_cond_drop_prob=0.1,
        text_cond_drop_prob=0.5,
        condition_on_text_encodings=False,  # set this to True if you wish to condition on text during training and sampling
    ).to(config.device)
