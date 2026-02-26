import math
import torch
import torch.nn as nn
import torch.special as special
import torch.amp as amp
from torch.utils.checkpoint import checkpoint

from functools import partial
from itertools import groupby

# helpers
from makani.models.common import DropPath, LayerScale, MLP, EncoderDecoder, SpectralConv
from makani.utils.features import get_water_channels, get_channel_groups

# get spectral transforms and spherical convolutions from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# get pre-formulated layers
#from makani.models.common import GeometricInstanceNormS2
from makani.mpu.layers import DistributedMLP, DistributedEncoderDecoder

# more distributed stuff
from makani.utils import comm

# layer normalization
from physicsnemo.distributed.mappings import scatter_to_parallel_region, gather_from_parallel_region
from physicsnemo.core import ModelMetaData
from physicsnemo.core import Module
from dataclasses import dataclass

# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    theta_cutoff_factor = {"piecewise linear": 0.5, "morlet": 0.5, "zernike": math.sqrt(2.0)}

    return (kernel_shape[0] + 1) * theta_cutoff_factor[basis_type] * math.pi / float(nlat - 1)

# commenting out torch.compile due to long intiial compile times
# @torch.compile
def _soft_clamp(x: torch.Tensor, offset: float = 0.0):
    x = x + offset
    y = torch.where(x > 0.0, x**2, 0.0)
    y = torch.where(x >= 0.5, x - 0.25, y)
    return y


class DiscreteContinuousEncoder(nn.Module):
    def __init__(
        self,
        inp_shape=(721, 1440),
        out_shape=(480, 960),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
        out_chans=2,
        kernel_shape=(3,3),
        basis_type="morlet",
        basis_norm_mode="mean",
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        groups=1,
        bias=False,
        theta_cutoff_factor=None,
    ):
        super().__init__()

        # heuristic for finding theta_cutoff
        theta_cutoff = _compute_cutoff_radius(nlat=inp_shape[0], kernel_shape=kernel_shape, basis_type=basis_type) * theta_cutoff_factor

        # set up local convolution
        conv_handle = thd.DistributedDiscreteContinuousConvS2 if comm.get_size("spatial") > 1 else th.DiscreteContinuousConvS2
        self.conv = conv_handle(
            inp_chans,
            out_chans,
            in_shape=inp_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid_in,
            grid_out=grid_out,
            groups=groups,
            bias=bias,
            theta_cutoff=theta_cutoff,
        )
        if comm.get_size("spatial") > 1:
            self.conv.weight.is_shared_mp = ["spatial"]
            self.conv.weight.sharded_dims_mp = [None, None, None]
            if self.conv.bias is not None:
                self.conv.bias.is_shared_mp = ["spatial"]
                self.conv.bias.sharded_dims_mp = [None]

        if use_mlp:
            with torch.no_grad():
                self.conv.weight *= math.sqrt(2.0)

            self.act = activation_function()

            self.mlp = EncoderDecoder(
                num_layers=1,
                input_dim=out_chans,
                output_dim=out_chans,
                hidden_dim=int(mlp_ratio * out_chans),
                act_layer=activation_function,
                input_format="nchw",
            )

    def forward(self, x):
        dtype = x.dtype

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.conv(x)
            x = x.to(dtype=dtype)

        if hasattr(self, "act"):
            x = self.act(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        return x


class DiscreteContinuousDecoder(nn.Module):
    def __init__(
        self,
        inp_shape=(480, 960),
        out_shape=(721, 1440),
        grid_in="equiangular",
        grid_out="equiangular",
        inp_chans=2,
        out_chans=2,
        kernel_shape=(3, 3),
        basis_type="morlet",
        basis_norm_mode="mean",
        use_mlp=False,
        mlp_ratio=2.0,
        activation_function=nn.GELU,
        groups=1,
        bias=False,
        upsample_sht=False,
        upsample_shuffle=False,
        theta_cutoff_factor=None,
    ):
        super().__init__()

        if use_mlp:
            self.mlp = EncoderDecoder(
                num_layers=1, input_dim=inp_chans, output_dim=inp_chans, hidden_dim=int(mlp_ratio * inp_chans), act_layer=activation_function, input_format="nchw", gain=2.0
            )

            self.act = activation_function()

        # init distributed torch-harmonics if needed
        if comm.get_size("spatial") > 1:
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)

        # spatial parallelism in the SHT
        self.linear = None
        if upsample_sht:
            # set up sht for upsampling
            sht_handle = thd.DistributedRealSHT if comm.get_size("spatial") > 1 else th.RealSHT
            isht_handle = thd.DistributedInverseRealSHT if comm.get_size("spatial") > 1 else th.InverseRealSHT

            # set upsampling module
            self.sht = sht_handle(*inp_shape, grid=grid_in).float()
            self.isht = isht_handle(*out_shape, lmax=self.sht.lmax, mmax=self.sht.mmax, grid=grid_out).float()
            self.upsample = nn.Sequential(self.sht, self.isht)
        elif upsample_shuffle:
            scale_factor = out_shape[0] // inp_shape[0]
            resample_handle = thd.DistributedResampleS2 if comm.get_size("spatial") > 1 else th.ResampleS2
            self.upsample = nn.Sequential(
                nn.Conv2d(inp_chans, inp_chans // (scale_factor**2), kernel_size=1, bias=False), 
                resample_handle(*inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out, mode="bilinear")
            )
            inp_chans = inp_chans // (scale_factor**2)
        else:
            resample_handle = thd.DistributedResampleS2 if comm.get_size("spatial") > 1 else th.ResampleS2
            self.upsample = resample_handle(*inp_shape, *out_shape, grid_in=grid_in, grid_out=grid_out, mode="bilinear")

        # heuristic for finding theta_cutoff
        # nto entirely clear if out or in shape should be used here with a non-conv method for upsampling
        theta_cutoff = _compute_cutoff_radius(nlat=out_shape[0], kernel_shape=kernel_shape, basis_type=basis_type) * theta_cutoff_factor

        # set up DISCO convolution
        conv_handle = thd.DistributedDiscreteContinuousConvS2 if comm.get_size("spatial") > 1 else th.DiscreteContinuousConvS2
        self.conv = conv_handle(
            inp_chans,
            out_chans,
            in_shape=out_shape,
            out_shape=out_shape,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            basis_norm_mode=basis_norm_mode,
            grid_in=grid_out,
            grid_out=grid_out,
            groups=groups,
            bias=False,
            theta_cutoff=theta_cutoff,
        )
        if comm.get_size("spatial") > 1:
            self.conv.weight.is_shared_mp = ["spatial"]
            self.conv.weight.sharded_dims_mp = [None, None, None]
            if self.conv.bias is not None:
                self.conv.bias.is_shared_mp = ["spatial"]
                self.conv.bias.sharded_dims_mp = [None]

    def forward(self, x):
        dtype = x.dtype

        if hasattr(self, "act"):
            x = self.act(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        with amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            x = self.upsample(x)
            x = self.conv(x)
            x = x.to(dtype=dtype)
        return x


class NeuralOperatorBlock(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        inp_chans,
        out_chans,
        conv_type="local",
        mlp_ratio=2.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.Identity,
        num_groups=1,
        skip="identity",
        layer_scale=True,
        use_mlp=False,
        kernel_shape=(3, 3),
        basis_type="morlet",
        basis_norm_mode="mean",
        checkpointing_level=0,
        bias=False,
        theta_cutoff_factor=2.0,
    ):
        super().__init__()

        # determine some shapes
        self.inp_shape = (forward_transform.nlat, forward_transform.nlon)
        self.out_shape = (inverse_transform.nlat, inverse_transform.nlon)
        self.out_chans = out_chans

        # gain factor for the convolution
        gain_factor = 1.0

        # disco convolution layer
        if conv_type == "local":

            # heuristic for finding theta_cutoff
            theta_cutoff = _compute_cutoff_radius(nlat=self.inp_shape[0], kernel_shape=kernel_shape, basis_type=basis_type) * theta_cutoff_factor

            conv_handle = thd.DistributedDiscreteContinuousConvS2 if comm.get_size("spatial") > 1 else th.DiscreteContinuousConvS2
            self.local_conv = conv_handle(
                inp_chans,
                inp_chans,
                in_shape=self.inp_shape,
                out_shape=self.out_shape,
                kernel_shape=kernel_shape,
                basis_type=basis_type,
                basis_norm_mode=basis_norm_mode,
                groups=num_groups,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=False,
                theta_cutoff=theta_cutoff,
            )
            if comm.get_size("spatial") > 1:
                self.local_conv.weight.is_shared_mp = ["spatial"]
                self.local_conv.weight.sharded_dims_mp = [None, None, None]
                if self.local_conv.bias is not None:
                    self.local_conv.bias.is_shared_mp = ["spatial"]
                    self.local_conv.bias.sharded_dims_mp = [None]

            with torch.no_grad():
                self.local_conv.weight *= gain_factor

        elif conv_type == "global":
            # convolution layer
            self.global_conv = SpectralConv(
                forward_transform,
                inverse_transform,
                inp_chans,
                inp_chans,
                operator_type="dhconv",
                num_groups=num_groups,
                bias=bias,
                gain=gain_factor,
            )
        else:
            raise ValueError(f"Unknown convolution type {conv_type}")

        # norm layer
        self.norm = norm_layer()

        if use_mlp == True:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(inp_chans * mlp_ratio)
            self.mlp = MLPH(
                in_features=inp_chans,
                out_features=out_chans,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                drop_type="features",
                checkpointing=(checkpointing_level >= 2),
                gain=gain_factor,
            )

        # dropout
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        if layer_scale:
            self.layer_scale = LayerScale(out_chans)
            self.layer_scale.weight.is_shared_mp = ["spatial"]
            self.layer_scale.weight.sharded_dims_mp = [None, None, None, None]
        else:
            self.layer_scale = nn.Identity()

        # skip connection
        if skip == "linear":
            gain_factor = 1.0
            self.skip = nn.Conv2d(inp_chans, out_chans, 1, 1, bias=False)
            torch.nn.init.normal_(self.skip.weight, std=math.sqrt(gain_factor / inp_chans))
            self.skip.weight.is_shared_mp = ["spatial"]
            self.skip.weight.sharded_dims_mp = [None, None, None, None]
            if self.skip.bias is not None:
                self.skip.bias.is_shared_mp = ["spatial"]
                self.skip.bias.sharded_dims_mp = [None]
        elif skip == "identity":
            self.skip = nn.Identity()
        elif skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {skip}")

    def forward(self, x):
        """
        Updated NO block
        """

        if hasattr(self, "global_conv"):
            dx, _ = self.global_conv(x)
        elif hasattr(self, "local_conv"):
            dx = self.local_conv(x)

        if hasattr(self, "norm"):
            dx = self.norm(dx)

        if hasattr(self, "mlp"):
            dx = self.mlp(dx)

        dx = self.drop_path(dx)

        if hasattr(self, "skip"):
            x = self.skip(x[..., : self.out_chans, :, :]) + self.layer_scale(dx)
        else:
            x = dx

        return x

@dataclass
class MetaData(ModelMetaData):
    name: str = "PrecipNeuralOperatorNet"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class PrecipNeuralOperatorNet(Module):
    """
    Backbone of the FourCastNet2 architecture. Uses a Spherical Neural Operator which is derived from the
    Spherical Fourier Neural Operator and augmented with localized spherical Neural Operator Convolutions.
    Encoder and Decoder are grouped into channel groups to treat armospheric and surface variables appropriately.

    References:
    [1] Bonev et al., Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
    [2] Ocampo et al., Scalable and Equivariant Spherical CNNs by Discrete-Continuous (DISCO) Convolutions
    [3] Liu-Schiaffini et al., Neural Operators with Localized Integral and Differential Kernels
    """

    def __init__(
        self,
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        in_channels=27,
        out_channels=1,
        kernel_shape=(3, 3),
        filter_basis_type="morlet",
        filter_basis_norm_mode="mean",
        scale_factor=8,
        encoder_mlp=False,
        upsample_sht=False,
        upsample_shuffle=False,
        channel_names=["u500", "v500"],
        n_history=0,
        embed_dim=8,
        noise_embed_dim=8,
        num_layers=4,
        num_groups=1,
        use_mlp=True,
        mlp_ratio=2.0,
        activation_function="gelu",
        layer_scale=True,
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="none",
        max_modes=None,
        hard_thresholding_fraction=1.0,
        big_skip=False,
        bias=False,
        checkpointing_level=0,
        encoder_theta_cutoff_factor=1.0,
        decoder_theta_cutoff_factor=1.0,
        theta_cutoff_factor=1.0,
        use_encoder: bool = True,
        use_decoder: bool = True,
        head_mlp_ratio: float = None,
        block_conv_type: list = ["local"] * 4,
        noise_mode: str = None,
        noise_channels: int = 1,
        **kwargs,
    ):
        super().__init__(meta=MetaData())

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.embed_dim = embed_dim
        self.big_skip = big_skip
        self.checkpointing_level = checkpointing_level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_encoder = use_encoder
        self.use_decoder = use_decoder
        self.noise_mode = noise_mode
        self.noise_channels = int(noise_channels)
        self.noise_embed_dim = noise_embed_dim
        self.upsample_shuffle = upsample_shuffle
        # currently doesn't support neither history nor future:
        assert n_history == 0

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        # initialize spectral transforms
        self._init_spectral_transforms(model_grid_type, sht_grid_type, hard_thresholding_fraction, max_modes)

        # Either treat channels uniformly or split into groups
        self.embed_dim = embed_dim
        
        # convert kernel shape to tuple
        kernel_shape = tuple(kernel_shape)

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")
        
        if self.use_encoder:
            self.encoder = DiscreteContinuousEncoder(
                inp_shape=tuple(inp_shape),
                out_shape=(self.h, self.w),
                inp_chans=self.in_channels,
                out_chans=self.embed_dim,
                grid_in=model_grid_type,
                grid_out=sht_grid_type,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                basis_norm_mode=filter_basis_norm_mode,
                activation_function=activation_function,
                groups=1,
                bias=bias,
                use_mlp=encoder_mlp,
                theta_cutoff_factor=encoder_theta_cutoff_factor,
            )
            if self.noise_mode is not None:
                # Encode noise to feature space (embed_dim). This keeps downstream shapes stable.
                self.noise_encoder = DiscreteContinuousEncoder(
                    inp_shape=tuple(inp_shape),
                    out_shape=(self.h, self.w),
                    inp_chans=self.noise_channels,
                    out_chans=self.noise_embed_dim,
                    grid_in=model_grid_type,
                    grid_out=sht_grid_type,
                    kernel_shape=kernel_shape,
                    basis_type=filter_basis_type,
                    basis_norm_mode=filter_basis_norm_mode,
                    activation_function=activation_function,
                    groups=1,
                    bias=bias,
                    use_mlp=encoder_mlp,
                    theta_cutoff_factor=encoder_theta_cutoff_factor,
                )
                if self.noise_mode == "concatenate":
                    self.noise_mergers = nn.ModuleList([])
                    for _ in range(num_layers):
                        self.noise_mergers.append(nn.Conv2d(self.embed_dim + self.noise_embed_dim, self.embed_dim, kernel_size=1, bias=False))
                        if hasattr(self.noise_mergers[-1], "weight") and hasattr(self.noise_mergers[-1].weight, "is_shared_mp"):
                            self.noise_mergers[-1].weight.is_shared_mp = ["spatial"]
                            self.noise_mergers[-1].weight.sharded_dims_mp = [None, None, None, None]
        else:
            self.encoder = nn.Identity()

        # decoder
        if self.use_decoder:
            self.decoder = DiscreteContinuousDecoder(
                inp_shape=(self.h, self.w),
                out_shape=tuple(inp_shape),
                inp_chans=self.embed_dim,
                out_chans=self.out_channels,
                grid_in=sht_grid_type,
                grid_out=model_grid_type,
                kernel_shape=kernel_shape,
                basis_type=filter_basis_type,
                basis_norm_mode=filter_basis_norm_mode,
                activation_function=activation_function,
                groups=math.gcd(self.out_channels, self.embed_dim),
                bias=bias,
                use_mlp=encoder_mlp,
                upsample_sht=upsample_sht,
                upsample_shuffle=upsample_shuffle,
                theta_cutoff_factor=decoder_theta_cutoff_factor,
            )
        else:
            self.decoder = nn.Identity()

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # get the handle for the normalization layer
        norm_layer = self._get_norm_layer_handle(self.h, self.w, self.embed_dim, normalization_layer=normalization_layer, sht_grid_type=sht_grid_type)

        # Internal NO blocks (skip when encoder-only)
        self.blocks = nn.ModuleList([])
        if num_layers > 0:
            for i in range(num_layers):
                first_layer = i == 0
                last_layer = i == num_layers - 1
                
                if first_layer and not self.use_encoder:
                    in_ch = self.in_channels
                else:
                    in_ch = self.embed_dim
                
                if last_layer and not self.use_decoder:
                    out_ch = self.out_channels
                else:
                    out_ch = self.embed_dim

                block = NeuralOperatorBlock(
                    self.sht,
                    self.isht,
                    in_ch,
                    out_ch,
                    conv_type=block_conv_type[i],
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    path_drop_rate=dpr[i],
                    act_layer=activation_function,
                    norm_layer=norm_layer,
                    skip="identity",
                    layer_scale=layer_scale,
                    use_mlp=use_mlp,
                    kernel_shape=kernel_shape,
                    basis_type=filter_basis_type,
                    basis_norm_mode=filter_basis_norm_mode,
                    bias=bias,
                    checkpointing_level=checkpointing_level,
                    theta_cutoff_factor=theta_cutoff_factor,
                )

                self.blocks.append(block)

        # residual prediction
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False)
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            if self.residual_transform.bias is not None:
                self.residual_transform.bias.is_shared_mp = ["spatial"]
                self.residual_transform.bias.sharded_dims_mp = [None]
            scale = math.sqrt(0.5 / max(1, self.in_channels))
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)


    @torch.compiler.disable(recursive=False)
    def _init_spectral_transforms(
        self,
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        hard_thresholding_fraction=1.0,
        max_modes=None,
    ):
        """
        Initialize the spectral transforms based on the maximum number of modes to keep. Handles the computation
        of local image shapes and domain parallelism, based on the
        """

        # precompute the cutoff frequency on the sphere
        if max_modes is not None:
            modes_lat, modes_lon = max_modes
        else:
            modes_lat = int(self.h * hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * hard_thresholding_fraction)

        sht_handle = th.RealSHT
        isht_handle = th.InverseRealSHT

        # spatial parallelism in the SHT
        if comm.get_size("spatial") > 1:
            polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
            azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
            thd.init(polar_group, azimuth_group)
            sht_handle = thd.DistributedRealSHT
            isht_handle = thd.DistributedInverseRealSHT

        # set up
        self.sht = sht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()
        self.isht = isht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()

    @torch.compiler.disable(recursive=True)
    def _get_norm_layer_handle(
        self,
        h,
        w,
        embed_dim,
        normalization_layer="none",
        sht_grid_type="legendre-gauss",
    ):
        """
        get the handle for ionitializing normalization layers
        """
        # pick norm layer
        if normalization_layer == "layer_norm":
            from makani.mpu.layer_norm import DistributedLayerNorm
            norm_layer_handle = partial(DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6)
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                from makani.mpu.layer_norm import DistributedInstanceNorm2d
                norm_layer_handle = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            else:
                norm_layer_handle = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif normalization_layer == "instance_norm_s2":
            if comm.get_size("spatial") > 1:
                from makani.mpu.layer_norm import DistributedGeometricInstanceNormS2
                norm_layer_handle = DistributedGeometricInstanceNormS2
            else:
                from makani.models.common import GeometricInstanceNormS2
                norm_layer_handle = GeometricInstanceNormS2
            norm_layer_handle = partial(
                norm_layer_handle,
                img_shape=(h, w),
                crop_shape=(h, w),
                crop_offset=(0, 0),
                grid_type=sht_grid_type,
                pole_mask=0,
                num_features=embed_dim,
                eps=1e-6,
                affine=True,
            )
        elif normalization_layer == "none":
            norm_layer_handle = nn.Identity
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

        return norm_layer_handle

    def encode(self, x, noise=None):
        """
        forward pass for the encoder
        """
        x_out = self.encoder(x.contiguous())
        n = None
        if (noise is not None) and (self.noise_mode is not None):
            n = self.noise_encoder(noise.contiguous())
        return x_out, n

    def decode(self, x):
        """
        forward pass for the decoder
        """
        x_out = self.decoder(x)
        return x_out

    def processor_blocks(self, x, noise=None):
        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        for i, blk in enumerate(self.blocks):
            if self.noise_mode == "add" and noise is not None:
                x = x + noise
            elif self.noise_mode == "concatenate" and noise is not None:
                x = torch.cat([x, noise], dim=1)
                x = self.noise_mergers[i](x)
            # append the auxiliary channels to the input of each block
            if self.checkpointing_level >= 3:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return x

    def forward(self, x, noise=None):
        # save big skip residual from the original input
        if self.big_skip:
            residual = x.contiguous()

        # run the encoder
        if self.checkpointing_level >= 1:
            x, noise = checkpoint(self.encode, x, noise, use_reentrant=False)
        else:
            x, noise = self.encode(x, noise)

        # run the processor
        x = self.processor_blocks(x, noise)

        # run the decoder
        if self.checkpointing_level >= 1:
            x = checkpoint(self.decode, x, use_reentrant=False)
        else:
            x = self.decode(x)

        if self.big_skip:
            x = x + self.residual_transform(residual)

        return x