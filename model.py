import torch
import torch.nn as nn
from EDM_nets import DhariwalUNet, SongUNet

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1

        labels = torch.where(drop_ids.to(labels.device),
                             self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class UDPM_Net(nn.Module):
    def __init__(self, in_shape, model_channels, channel_mult, attn_resolutions, num_blocks_per_res=4, classes_num=0,
                 in_channels=3, out_channels=3, class_dropout_prob=0, for_SR=False, net_type='EDM', dropout=0, sf=1):
        super(UDPM_Net, self).__init__()
        self.in_shape = in_shape
        self.classes_num = classes_num
        self.class_dropout_prob = class_dropout_prob
        self.for_SR = for_SR
        self.sf = sf
        self.net_type = net_type
        if net_type == 'Dhariwal':
            self.net = DhariwalUNet(img_resolution      = in_shape,                     # Image resolution at input/output.
                                    in_channels         = in_channels,                        # Number of color channels at input.
                                    out_channels        = out_channels * sf ** 2,                       # Number of color channels at output.
                                    label_dim           = classes_num + (class_dropout_prob > 0),            # Number of class labels, 0 = unconditional.
                                    augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

                                    model_channels      = model_channels,          # Base multiplier for the number of channels.
                                    channel_mult        = channel_mult,    # Per-resolution multipliers for the number of channels.
                                    channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
                                    num_blocks          = num_blocks_per_res,            # Number of residual blocks per resolution.
                                    attn_resolutions    = attn_resolutions,    # List of resolutions with self-attention.
                                    dropout             = dropout,
                                    label_dropout       = class_dropout_prob, # Dropout probability of class labels for classifier-free guidance.
            )
        elif net_type == 'NCSN':
            self.net = SongUNet(
                                    img_resolution      = in_shape,                     # Image resolution at input/output.
                                    in_channels         = in_channels,                        # Number of color channels at input.
                                    out_channels        = out_channels * sf ** 2,                       # Number of color channels at output.
                                    label_dim           = classes_num + (class_dropout_prob > 0),            # Number of class labels, 0 = unconditional.
                                    augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

                                    model_channels      = model_channels,          # Base multiplier for the number of channels.
                                    channel_mult        = channel_mult,    # Per-resolution multipliers for the number of channels.
                                    channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
                                    num_blocks          = num_blocks_per_res,            # Number of residual blocks per resolution.
                                    attn_resolutions    = attn_resolutions,         # List of resolutions with self-attention.
                                    dropout             = dropout,         # Dropout probability of intermediate activations.
                                    label_dropout       = class_dropout_prob,            # Dropout probability of class labels for classifier-free guidance.

                                    embedding_type      = 'fourier', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                                    channel_mult_noise  = 2,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                                    encoder_type        = 'residual',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                                    decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                                    resample_filter     = [1,3,3,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
            )
        elif net_type == 'DDPM':
            self.net = SongUNet(
                                    img_resolution      = in_shape,                     # Image resolution at input/output.
                                    in_channels         = in_channels,                        # Number of color channels at input.
                                    out_channels        = out_channels * sf ** 2,                       # Number of color channels at output.
                                    label_dim           = classes_num + (class_dropout_prob > 0),            # Number of class labels, 0 = unconditional.
                                    augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

                                    model_channels      = model_channels,          # Base multiplier for the number of channels.
                                    channel_mult        = channel_mult,    # Per-resolution multipliers for the number of channels.
                                    channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
                                    num_blocks          = num_blocks_per_res,            # Number of residual blocks per resolution.
                                    attn_resolutions    = attn_resolutions,         # List of resolutions with self-attention.
                                    dropout             = dropout,         # Dropout probability of intermediate activations.
                                    label_dropout       = class_dropout_prob,            # Dropout probability of class labels for classifier-free guidance.

                                    embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                                    channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                                    encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                                    decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                                    resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
            )


    def forward(self, x, t, y=None):
        if len(t.shape) > 1:
            t = t.reshape(-1)

        if self.for_SR:
            x = self.up_layer(x)
        if self.classes_num > 1:
            out = self.net(x, t, torch.nn.functional.one_hot(y, self.classes_num + (self.class_dropout_prob > 0)).float())
        else:
            out = self.net(x, t, None)

        if self.sf > 1:
            out = nn.functional.pixel_shuffle(out, self.sf)
        return  out
