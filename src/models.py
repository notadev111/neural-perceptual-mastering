class MasteringModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ENCODER: TCN (dilated CNN)
        self.encoder = TCNEncoder(
            latent_dim=512,
            num_layers=8,  # Deeper for better features
            kernel_size=3,
            dilation_base=2
        )
        
        # PARAMETRIC PATH: Adaptive EQ (5-10 bands)
        self.parametric_decoder = AdaptiveParametricEQ(
            latent_dim=512,
            max_bands=10,  # Let model choose 5-10
            sample_rate=44100
        )
        
        # RESIDUAL PATH: Hybrid grey-box + black-box
        self.residual_decoder = HybridResidualDecoder(
            latent_dim=512,
            use_compressor=True,  # Grey-box
            use_saturator=True,   # Grey-box
            use_neural_residual=True  # Black-box catch-all
        )
    
    def forward(self, audio):
        # Encode
        z = self.encoder(audio)
        
        # Parametric EQ
        eq_out, eq_params = self.parametric_decoder(z, audio)
        
        # Residual processing
        residual_out, residual_params = self.residual_decoder(z, audio)
        
        # Combine
        output = eq_out + residual_out
        
        return output, {
            'eq_params': eq_params,
            'residual_params': residual_params
        }