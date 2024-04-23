class QuantConfig():
    def __init__(self,N_weight,N_ADC,array_size) -> None:
        self.N_weight=N_weight
        self.N_ADC=N_ADC
        self.array_size=array_size

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