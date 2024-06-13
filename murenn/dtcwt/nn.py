import torch
import murenn
import math

from .utils import fix_length


class MuReNNDirect(torch.nn.Module):
    """
    Args:
        J (int): Number of levels (octaves) in the DTCWT decomposition.
        Q (int): Number of Conv1D filters per octave.
        in_channels (int): Number of channels in the input signal.
        padding_mode (str): One of 'symmetric' (default), 'zeros', 'replicate',
            and 'circular'. Padding scheme for the DTCWT decomposition.
    """
    def __init__(self, *, J, Q, T, in_channels, padding_mode="symmetric"):
        super().__init__()
        self.Q = Q
        self.C = in_channels
        down = []
        conv1d = []
        self.dtcwt = murenn.DTCWT(
            J=J,
            padding_mode=padding_mode,
        )

        for j in range(J):
            down_j = murenn.DTCWT(
                J=J-j,
                padding_mode=padding_mode,
                skip_hps=True,
                normalize=False,
            )
            down.append(down_j)

            conv1d_j = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=Q*in_channels,
                kernel_size=T,
                bias=False,
                groups=in_channels,
                padding="same",
            )
            torch.nn.init.normal_(conv1d_j.weight)
            conv1d.append(conv1d_j)

        self.down = torch.nn.ModuleList(down)
        self.conv1d = torch.nn.ParameterList(conv1d)


    def forward(self, x):
        """
        Args:
            x (PyTorch tensor): A tensor of shape `(B, in_channels, T)`. B is a batch size. 
                in_channels is the number of channels in the input tensor, this should match
                the in_channels attribute of the class instance. T is a length of signal sequence. 
        Returns:
            y (PyTorch tensor): A tensor of shape `(B, in_channels, Q, J, T_out)`
        """
        assert self.C == x.shape[1]
        lp, bps = self.dtcwt(x)
        output = []
        for j in range(self.dtcwt.J):
            Wx_j_r = self.conv1d[j](bps[j].real)
            Wx_j_i = self.conv1d[j](bps[j].imag)
            UWx_j = ModulusStable.apply(Wx_j_r, Wx_j_i)
            UWx_j, _ = self.down[j](UWx_j)
            B, _, N = UWx_j.shape
            # reshape from (B, C*Q, N) to (B, C, Q, N)
            UWx_j = UWx_j.view(B, self.C, self.Q, N)
            output.append(UWx_j)
        return torch.stack(output, dim=3)
    
    @property
    def to_conv1d(self):
        """
        Get the per scale, per filter, per input channel impulse responses.
        -------
        Return:
            y (PyTorch tensor): A complex-valued tensor of shape `(B, in_channels*J*Q, (2**J)*Q)`
        """
        # T the filter length
        T = self.conv1d[0].kernel_size[0]
        # J the number of levels of decompostion
        J = self.dtcwt.J
        # Generate the impulse signal, this signal is zero padded to a length of (2**J)*T
        N = 2**J * T
        x = torch.zeros(1, self.C, N)
        x[:, :, N//2] = 1
        # Get the padding mode
        padding_mode = self.dtcwt.padding_mode
        if padding_mode == "constant":
            padding_mode = "zeros"

        inv = murenn.IDTCWT(
            J=J,
            padding_mode=padding_mode,
        )
        # Get DTCWT impulse reponses
        phi, psis = self.dtcwt(x)
        # Set phi to a zero valued tensor
        zeros_phi = phi.new_zeros(size=(1, self.C*self.Q, phi.shape[-1]))
        # Create an empty list for {w_jq}
        ws = []
        for j in range(J):
            # Wpsi_jr = Re[psi_j] * w_jq
            Wpsi_jr = self.conv1d[j](psis[j].real)
            # W_ji = Im[psi_j] * w_jq
            Wpsi_ji = self.conv1d[j](psis[j].imag)
            # Set the coefficients besides this scale to zero .repeat(ch, 1, 1)
            Wpsis_jr = [Wpsi_jr * (1 + 0j) if k == j else psis[k].new_zeros(size=psis[k].shape).repeat(1, self.Q, 1) for k in range(J)]
            Wpsis_ji = [Wpsi_ji * (0 + 1j) if k == j else psis[k].new_zeros(size=psis[k].shape).repeat(1, self.Q, 1) for k in range(J)]
            # Get the impulse response
            w_jr = inv(zeros_phi, Wpsis_jr)
            w_ji = inv(zeros_phi, Wpsis_ji)
            w_j = torch.complex(w_jr, w_ji)
            ws.append(w_j)
        ws = torch.cat(ws, dim=1)
        return ws


class ModulusStable(torch.autograd.Function):
    """Stable complex modulus

    This class implements a modulus transform for complex numbers which is
    stable with respect to very small inputs (z close to 0), avoiding
    returning NaN's in all cases.

    -------
    Adapted from Kymatio
    """
    @staticmethod
    def forward(ctx, x_r, x_i):
        output = (x_r ** 2 + x_i ** 2).sqrt()
        ctx.save_for_backward(x_r, x_i, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_r, x_i, output = ctx.saved_tensors
        dxr, dxi = None, None
        if ctx.needs_input_grad[0]:
            dxr = x_r.mul(grad_output).div(output)
            dxi = x_i.mul(grad_output).div(output)
            dxr.masked_fill_(output == 0, 0)
            dxi.masked_fill_(output == 0, 0)
        return dxr, dxi