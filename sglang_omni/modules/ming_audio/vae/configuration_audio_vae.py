from transformers import PretrainedConfig


class AudioVAEconfig(PretrainedConfig):
    def __init__(
        self,
        sample_rate: int = 16000,
        enc_kwargs: dict = None,
        dec_kwargs: dict = None,
        init_method="normal",
        patch_size=-1,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.enc_kwargs = enc_kwargs
        self.dec_kwargs = dec_kwargs
        self.init_method = init_method
        self.patch_size = patch_size
        super().__init__(**kwargs)
