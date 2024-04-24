class QuantConfig():
    def __init__(self, N_weight:int, N_ADC:int, array_size:int) -> None:
        self.N_weight=N_weight
        self.N_ADC=N_ADC
        self.array_size=array_size

class NoiseConfig():
    def __init__(self, noise_type:str, rate_max:float, rate_zero:float, N:int=1, m:int=1) -> None:
        self.noise_type=noise_type
        self.rate_max=rate_max
        self.rate_zero=rate_zero
        self.m=m

quant_config = {
    "MLP3": QuantConfig(
        N_weight=4,
        N_ADC=4,
        array_size=32
    ),
    "LeNet": QuantConfig(
        N_weight=4,
        N_ADC=4,
        array_size=32
    ),
    "CIFAR": QuantConfig(
        N_weight=6,
        N_ADC=6,
        array_size=64
    ),
}

noise_config = {
    "RRAM1": NoiseConfig(
        noise_type="Gaussian",
        rate_max=1,
        rate_zero=1,
        m=1,
    ),
    "FeFET2": NoiseConfig(
        noise_type="Four",
        rate_max=2,
        rate_zero=2,
        m = 2
    ),
    "RRAM4": NoiseConfig(
        noise_type="Four",
        rate_max=4,
        rate_zero=4,
        m=2
    ),
    "FeFET6": NoiseConfig(
        noise_type="Four",
        rate_max=6,
        rate_zero=6,
        m=2
    ),

}