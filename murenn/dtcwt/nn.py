import torch
import murenn
import math

from .utils import fix_length


class MuReNNDirect(torch.nn.Module):
    """
    Args:
        J (int): Number of levels (octaves) in the DTCWT decomposition.
        Q (int or list): Number of Conv1D filters per octave.
        T (int): Conv1D Kernel size multiplier. The Conv1d kernel size at scale j is equal to
          T * Q[j] where Q[j] is the number of filters.
        J_phi (int): Number of levels of downsampling. Stride is 2**J_phi. Default is J.
        in_channels (int): Number of channels in the input signal.
        padding_mode (str): One of 'symmetric' (default), 'zeros', 'replicate',
            and 'circular'. Padding scheme for the DTCWT decomposition.
    """
    def __init__(self, *, J, Q, T, in_channels, J_phi=None, padding_mode="symmetric"):
        super().__init__()
        if isinstance(Q, int):
            self.Q = [Q for j in range(J)]
        elif isinstance(Q, list):
            assert len(Q) == J
            self.Q = Q
        else:
            raise TypeError(f"Q must to be int or list, got {type(Q)}")
        if J_phi is None:
            J_phi = J
        if J_phi < J:
            raise ValueError("J_phi must be greater or equal to J")
        self.T = [T*self.Q[j] for j in range(J)]
        self.in_channels = in_channels
        self.padding_mode = padding_mode
        down = []
        conv1d = []
        self.dtcwt = murenn.DTCWT(
            J=J,
            padding_mode=padding_mode,
            alternate_gh=False,
        )

        for j in range(J):
            down_j = murenn.DTCWT(
                J=J_phi-j,
                padding_mode=padding_mode,
                skip_hps=True,
                alternate_gh=False,
            )
            down.append(down_j)

            conv1d_j = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.Q[j]*in_channels,
                kernel_size=self.T[j],
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
        assert self.in_channels == x.shape[1]
        lp, bps = self.dtcwt(x)
        UWx = []
        for j in range(self.dtcwt.J):
            Wx_j_r = self.conv1d[j](bps[j].real) + bps[j].real
            Wx_j_i = self.conv1d[j](bps[j].imag) + bps[j].imag
            UWx_j = ModulusStable.apply(Wx_j_r, Wx_j_i)
            # Avarange over time
            UWx_j, _ = self.down[j](UWx_j)
            B, _, N = UWx_j.shape
            UWx_j = UWx_j.view(B, self.in_channels, self.Q[j], N)
            UWx.append(UWx_j)
        UWx = torch.cat(UWx, dim=2)
        return UWx
    

    def to_conv1d(self):
        """
        Compute the single-resolution equivalent impulse response of the MuReNN layer.
        This would be helpful for visualization in Fourier domain, for receptive fields,
        and for comparing computational costs.
           DTCWT        conv1d        IDTCWT
        δ -------> ψ_j --------> w_jq -------> y_jq
        -------
        Return:
            conv1d (torch.nn.Conv1d): A Pytorch Conv1d instance with weights initialized to y_jq.
        """

        device = self.conv1d[0].weight.data.device
        # T the filter length
        T = max(self.T)
        # J the number of levels of decompostion
        J = self.dtcwt.J
        # Generate the impulse signal
        N = 2 ** J * T
        x = torch.zeros(1, self.in_channels, N).to(device)
        x[:, :, N//2] = 1
        inv = murenn.IDTCWT(
            J = J,
            alternate_gh=False         
        ).to(device)
        # Get DTCWT impulse reponses
        phi, psis = self.dtcwt(x)
        # Set phi to a zero valued tensor
        zeros_phi = phi.new_zeros(1, 1, phi.shape[-1])
        ws = []
        for j in range(J):
            Wpsi_jr = self.conv1d[j](psis[j].real).reshape(self.in_channels, self.Q[j], -1)
            Wpsi_ji = self.conv1d[j](psis[j].imag).reshape(self.in_channels, self.Q[j], -1)
            for q in range(self.Q[j]):
                Wpsi_jqr = Wpsi_jr[0, q, :].reshape(1,1,-1)
                Wpsi_jqi = Wpsi_ji[0, q, :].reshape(1,1,-1)
                Wpsis_r = [Wpsi_jqr * (1+0j) if k == j else psis[k].new_zeros(1,1,psis[k].shape[-1]) for k in range(J)]
                Wpsis_i = [Wpsi_jqi * (0+1j) if k == j else psis[k].new_zeros(1,1,psis[k].shape[-1]) for k in range(J)]
                w_r = inv(zeros_phi, Wpsis_r)
                w_i = inv(zeros_phi, Wpsis_i)
                ws.append(torch.complex(w_r, w_i))
        ws = torch.cat(ws, dim=0)
        conv1d = torch.nn.Conv1d(
            in_channels=1,
            out_channels=ws.shape[0],
            kernel_size=N,
            bias=False,
            padding="same",
        )
        conv1d.weight.data = torch.nn.parameter.Parameter(ws)
        return conv1d


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