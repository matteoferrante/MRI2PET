from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

#from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Reshape
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.utils.deprecate_utils import deprecated_arg
from monai.networks.layers.factories import Conv
from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding

from typing import Optional, Sequence, Tuple, Union
import torchbnn as bnn

"""What I will change here: I will introduce the chance to use depthwise separable convolutions instead of convolutions inside the ResidualUnits, becasue monai Classifier inherits from Regressor which includes Residual Units modules"""


class DepthSepConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        
        super().__init__()
        
        depth_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,stride=stride,padding=padding)
        point_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
    def forward(self,x):
        return self.depthwise_separable_conv(x)



class Convolution(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::

        -- (Conv|ConvTrans) -- (Norm -- Dropout -- Acti) --

    if ``conv_only`` set to ``True``::

        -- (Conv|ConvTrans) --

    For example:

    .. code-block:: python

        from monai.networks.blocks import Convolution

        conv = Convolution(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="ADN",
            act=("prelu", {"init": 0.2}),
            dropout=0.1,
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(conv)

    output::

        Convolution(
          (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (adn): ADN(
            (A): PReLU(num_parameters=1)
            (D): Dropout(p=0.1, inplace=False)
            (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
          )
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no no larger than the value of `spatial_dims`.
        dilation: dilation rate. Defaults to 1.
        groups: controls the connections between inputs and outputs. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only: whether to use the convolutional layer only. Defaults to False.
        is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
        output_padding: controls the additional size added to one side of the output shape.
            Defaults to None.

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    See also:

        :py:class:`monai.networks.layers.Conv`
        :py:class:`monai.networks.blocks.ADN`

    """

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
        dimensions: Optional[int] = None,
        depthwise: bool =False,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        self.depthwise=depthwise
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.dimensions]

        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
            
        elif depthwise:
            conv= DepthSepConv3d(in_channels,out_channels,kernel_size=kernel_size,stride=strides,padding=padding)
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            
        

        self.add_module("conv", conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.dimensions,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )






class ResidualUnit(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    For example:

    .. code-block:: python

        from monai.networks.blocks import ResidualUnit

        convs = ResidualUnit(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="AN",
            act=("prelu", {"init": 0.2}),
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(convs)

    output::

        ResidualUnit(
          (conv): Sequential(
            (unit0): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
            (unit1): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (residual): Identity()
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no no larger than the value of `dimensions`.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    See also:

        :py:class:`monai.networks.blocks.Convolution`

    """

    @deprecated_arg(name="dimensions", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        dimensions: Optional[int] = None,
        depthwise: bool =False,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                self.dimensions,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
                depthwise=depthwise
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = Conv[Conv.CONV, self.dimensions]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output




class Regressor(nn.Module):
    """
    This defines a network for relating large-sized input tensors to small output tensors, ie. regressing large
    values to a prediction. An output of a single dimension can be used as value regression or multi-label
    classification prediction, an output of a single value can be used as a discriminator or critic prediction.

    The network is constructed as a sequence of layers, either :py:class:`monai.networks.blocks.Convolution` or
    :py:class:`monai.networks.blocks.ResidualUnit`, with a final fully-connected layer resizing the output from the
    blocks to the final size. Each block is defined with a stride value typically used to downsample the input using
    strided convolutions. In this way each block progressively condenses information from the input into a deep
    representation the final fully-connected layer relates to a final result.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        out_shape: tuple of integers stating the dimension of the final output tensor (minus batch dimension)
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component

    Examples::

        # infers a 2-value result (eg. a 2D cartesian coordinate) from a 64x64 image
        net = Regressor((1, 64, 64), (2,), (2, 4, 8), (2, 2, 2))

    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
        depthwise: bool= False,
    ) -> None:
        super().__init__()

        self.in_channels, *self.in_shape = ensure_tuple(in_shape)
        self.dimensions = len(self.in_shape)
        self.channels = ensure_tuple(channels)
        self.strides = ensure_tuple(strides)
        self.out_shape = ensure_tuple(out_shape)
        self.kernel_size = ensure_tuple_rep(kernel_size, self.dimensions)
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.depthwise=depthwise
        self.net = nn.Sequential()
        
        echannel = self.in_channels

        padding = same_padding(kernel_size)

        self.final_size = np.asarray(self.in_shape, dtype=int)
        self.reshape = Reshape(*self.out_shape)

        # encode stage
        for i, (c, s) in enumerate(zip(self.channels, self.strides)):
            layer = self._get_layer(echannel, c, s, i == len(channels) - 1)
            echannel = c  # use the output channel number as the input for the next loop
            self.net.add_module("layer_%i" % i, layer)
            self.final_size = calculate_out_shape(self.final_size, kernel_size, s, padding)  # type: ignore

        self.final = self._get_final_layer((echannel,) + self.final_size)

    def _get_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> Union[ResidualUnit, Convolution]:
        """
        Returns a layer accepting inputs with `in_channels` number of channels and producing outputs of `out_channels`
        number of channels. The `strides` indicates downsampling factor, ie. convolutional stride. If `is_last`
        is True this is the final layer and is not expected to include activation and normalization layers.
        """

        layer: Union[ResidualUnit, Convolution]

        if self.num_res_units > 0:
            layer = ResidualUnit(
                subunits=self.num_res_units,
                last_conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                depthwise=self.depthwise
            )
        else:
            layer = Convolution(
                conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                depthwise=self.depthwise
            )

        return layer

    def _get_final_layer(self, in_shape: Sequence[int]):
        linear = nn.Linear(int(np.product(in_shape)), int(np.product(self.out_shape)))
        return nn.Sequential(nn.Flatten(), linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.final(x)
        x = self.reshape(x)
        return x
    
    
    
class BayesianRegressor(nn.Module):
    """
    Modification: The output is a Bayesian Linear Layer using torchbnn
    This defines a network for relating large-sized input tensors to small output tensors, ie. regressing large
    values to a prediction. An output of a single dimension can be used as value regression or multi-label
    classification prediction, an output of a single value can be used as a discriminator or critic prediction.

    The network is constructed as a sequence of layers, either :py:class:`monai.networks.blocks.Convolution` or
    :py:class:`monai.networks.blocks.ResidualUnit`, with a final fully-connected layer resizing the output from the
    blocks to the final size. Each block is defined with a stride value typically used to downsample the input using
    strided convolutions. In this way each block progressively condenses information from the input into a deep
    representation the final fully-connected layer relates to a final result.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        out_shape: tuple of integers stating the dimension of the final output tensor (minus batch dimension)
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component

    Examples::

        # infers a 2-value result (eg. a 2D cartesian coordinate) from a 64x64 image
        net = Regressor((1, 64, 64), (2,), (2, 4, 8), (2, 2, 2))

    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
        depthwise: bool= False,
        prior_mu: float = 0.,
        prior_sigma: float = 0.1,
        
        ) -> None:
        super().__init__()

        self.in_channels, *self.in_shape = ensure_tuple(in_shape)
        self.dimensions = len(self.in_shape)
        self.channels = ensure_tuple(channels)
        self.strides = ensure_tuple(strides)
        self.out_shape = ensure_tuple(out_shape)
        self.kernel_size = ensure_tuple_rep(kernel_size, self.dimensions)
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.depthwise=depthwise
        self.prior_mu=prior_mu
        self.prior_sigma=prior_sigma
        self.net = nn.Sequential()
        
        echannel = self.in_channels

        padding = same_padding(kernel_size)

        self.final_size = np.asarray(self.in_shape, dtype=int)
        self.reshape = Reshape(*self.out_shape)

        # encode stage
        for i, (c, s) in enumerate(zip(self.channels, self.strides)):
            layer = self._get_layer(echannel, c, s, i == len(channels) - 1)
            echannel = c  # use the output channel number as the input for the next loop
            self.net.add_module("layer_%i" % i, layer)
            self.final_size = calculate_out_shape(self.final_size, kernel_size, s, padding)  # type: ignore

        self.final = self._get_final_layer((echannel,) + self.final_size)

    def _get_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> Union[ResidualUnit, Convolution]:
        """
        Returns a layer accepting inputs with `in_channels` number of channels and producing outputs of `out_channels`
        number of channels. The `strides` indicates downsampling factor, ie. convolutional stride. If `is_last`
        is True this is the final layer and is not expected to include activation and normalization layers.
        """

        layer: Union[ResidualUnit, Convolution]

        if self.num_res_units > 0:
            layer = ResidualUnit(
                subunits=self.num_res_units,
                last_conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                depthwise=self.depthwise
            )
        else:
            layer = Convolution(
                conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                depthwise=self.depthwise
            )

        return layer

    def _get_final_layer(self, in_shape: Sequence[int]):
        linear=bnn.BayesLinear(prior_mu=self.prior_mu, prior_sigma=self.prior_sigma, in_features=int(np.product(in_shape)), out_features=int(np.product(self.out_shape)))
        #linear = nn.Linear(int(np.product(in_shape)), int(np.product(self.out_shape)))
        return nn.Sequential(nn.Flatten(), linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.final(x)
        x = self.reshape(x)
        return x
    
    
class Classifier(Regressor):
    """
    Defines a classification network from Regressor by specifying the output shape as a single dimensional tensor
    with size equal to the number of classes to predict. The final activation function can also be specified, eg.
    softmax or sigmoid.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        classes: integer stating the dimension of the final output tensor
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component
        last_act: name defining the last activation layer
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        classes: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
        last_act: Optional[str] = None,
        depthwise: bool=True,
    ) -> None:
        super().__init__(in_shape, (classes,), channels, strides, kernel_size, num_res_units, act, norm, dropout, bias,depthwise=depthwise)

        if last_act is not None:
            last_act_name, last_act_args = split_args(last_act)
            last_act_type = Act[last_act_name]

            self.final.add_module("lastact", last_act_type(**last_act_args))


            

    
class BayesianClassifier(BayesianRegressor):
    """
    Defines a classification network from Regressor by specifying the output shape as a single dimensional tensor
    with size equal to the number of classes to predict. The final activation function can also be specified, eg.
    softmax or sigmoid.

    Args:
        in_shape: tuple of integers stating the dimension of the input tensor (minus batch dimension)
        classes: integer stating the dimension of the final output tensor
        channels: tuple of integers stating the output channels of each convolutional layer
        strides: tuple of integers stating the stride (downscale factor) of each convolutional layer
        kernel_size: integer or tuple of integers stating size of convolutional kernels
        num_res_units: integer stating number of convolutions in residual units, 0 means no residual units
        act: name or type defining activation layers
        norm: name or type defining normalization layers
        dropout: optional float value in range [0, 1] stating dropout probability for layers, None for no dropout
        bias: boolean stating if convolution layers should have a bias component
        last_act: name defining the last activation layer
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        classes: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: Optional[float] = None,
        bias: bool = True,
        last_act: Optional[str] = None,
        depthwise: bool=True,
        prior_mu: float=0.,
        prior_sigma:float =0.1,
    ) -> None:
        super().__init__(in_shape, (classes,), channels, strides, kernel_size, num_res_units, act, norm, dropout, bias,depthwise=depthwise,prior_mu=prior_mu,prior_sigma=prior_sigma)

        if last_act is not None:
            last_act_name, last_act_args = split_args(last_act)
            last_act_type = Act[last_act_name]

            self.final.add_module("lastact", last_act_type(**last_act_args))