import torch
import murenn
import math

from .utils import fix_length


class MuReNNDirect(torch.nn.Module):
    """
    Args:
        J (int): Number of levels (octaves) in the DTCWT decomposition.
        Q (int or list): Number of Conv1D filters per octave.
        T (int): The Conv1d kernel size.
        in_channels (int): Number of channels in the input signal.
        J_phi (int): Number of levels of downsampling. Stride is 2**J_phi. Default is J-1.
        mu (float): Weighting factor for the indentity mapping. Default is 1.
        include_lp (bool): Whether to include the low-pass component in the output. Default is False.
        padding_mode (str): One of 'symmetric' (default), 'zeros', 'replicate',
            and 'circular'. Padding scheme for the DTCWT decomposition.
    """
    def __init__(self, *, J, Q, T, in_channels, J_phi=None, mu=1, include_lp=False, padding_mode="symmetric"):
        super().__init__()
        if isinstance(Q, int):
            self.Q = [Q for j in range(J)]
        elif isinstance(Q, list):
            assert len(Q) == J
            self.Q = Q
        else:
            raise TypeError(f"Q must to be int or list, got {type(Q)}")
        if J_phi is None:
            J_phi = J - 1
        if J_phi < (J - 1):
            raise ValueError("J_phi must be greater or equal to J-1")
        self.T = T
        self.in_channels = in_channels
        self.padding_mode = padding_mode
        self.mu = mu
        self.include_lp = include_lp
        down = []
        conv1d = []
        self.dtcwt = murenn.DTCWT(
            J=J,
            padding_mode=padding_mode,
        )

        for j in range(J):
            conv1d_j = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.Q[j],
                kernel_size=self.T,
                bias=False,
                padding="same",
            )
            torch.nn.init.normal_(conv1d_j.weight)
            conv1d.append(conv1d_j)
    
            down_j = Downsampling(J_phi - j)
            down.append(down_j)
        
        if self.include_lp:
            down.append(Downsampling(J_phi - J + 2))

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
            Wx_j_r = self.conv1d[j](bps[j].real)
            Wx_j_i = self.conv1d[j](bps[j].imag)
            UWx_j = ModulusStable.apply(Wx_j_r, Wx_j_i)
            UWx_j = self.down[j](UWx_j)
            # B, _, N = UWx_j.shape
            # UWx_j = UWx_j.view(B, self.in_channels, self.Q[j], N)
            UWx.append(UWx_j)
            
        if self.include_lp:
            UWx.append(self.down[-1](lp))

        UWx = torch.cat(UWx, dim=1)
        return UWx
    
    def to_conv1d(self):
        """
        Compute the single-resolution equivalent impulse response of the MuReNN layer.
        This would be helpful for visualization in Fourier domain, for receptive fields,
        and for comparing computational costs.
           conv1d        IDTCWT
        δ --------> w_jq -------> y_jq
        -------
        Return:
            conv1ds: A dictionary containing PyTorch Conv1d instances with weights initialized to y_jq.
                - "complex" (torch.nn.Conv1d): the equivalent complex hybrid filter
                - "real" (torch.nn.Conv1d): the real part of the hybrid filter
                - "imag" (torch.nn.Conv1d): the imaginary part of the hybrid filter
        """

        device = self.conv1d[0].weight.data.device
        T = self.T  # Filter length
        J = self.dtcwt.J  # Number of levels of decomposition
        N = 2 ** J * max(T, self.dtcwt.g0a.shape[-1]) * 2  # Hybrid filter length

        # Generate a zero signal
        x = torch.zeros(1, self.in_channels, N).to(device)

        # Initialize the inverse DTCWT
        inv = murenn.IDTCWT(J=J, alternate_gh=False).to(device)

        # Obtain two dual-tree response of the zero signal
        phi, psis = self.dtcwt(x)
        phi = phi[0,0,:].reshape(1,1,-1) # We only need the first channel

        ws_r, ws_i = [], []
        for j in range(J):
            # Set the level-j response to a impulse signal
            psi_j = psis[j].real
            psi_j[:, :, psi_j.shape[2]//2] = 1 / math.sqrt(2) ** j # The energy gain
            # Convolve the impulse signal with the conv1d filter
            Wpsi_j = self.conv1d[j](psi_j).reshape(1, self.Q[j], -1)
            # Apply dual-tree invert transform to obtain the hybrid wavelets.
            for q in range(self.Q[j]):
                Wpsi_jq = Wpsi_j[0, q, :].reshape(1,1,-1)
                Wpsis_r = [Wpsi_jq * (1+0j) if k == j else psis[k].new_zeros(1,1,psis[k].shape[-1]) for k in range(J)]
                Wpsis_i = [Wpsi_jq * (0+1j) if k == j else psis[k].new_zeros(1,1,psis[k].shape[-1]) for k in range(J)]

                ws_r.append(inv(phi, Wpsis_r))
                ws_i.append(inv(phi, Wpsis_i))
        
        ws_r = torch.cat(ws_r, dim=0)
        ws_i = torch.cat(ws_i, dim=0)

        def create_conv1d(weight):
            conv1d = torch.nn.Conv1d(
                in_channels=1,
                out_channels=weight.shape[0],
                kernel_size=N,
                bias=False,
                padding="same",
            )
            conv1d.weight.data = torch.nn.parameter.Parameter(weight)
            return conv1d

        return {
            "complex": create_conv1d(ws_r+1j*ws_i),
            "real": create_conv1d(ws_r),
            "imag": create_conv1d(ws_i),
        }


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


class Downsampling(torch.nn.Module):
    """
    Downsample the input signal by a factor of 2**J_phi.
    --------------------
    Args:
        J_phi (int): Number of levels of downsampling.
    """
    def __init__(self, J_phi):
        super().__init__()
        self.J_phi = J_phi
        # We are using a 13-tap low-pass filter
        self.phi = murenn.DTCWT(
            J=1,
            level1="near_sym_b",
            skip_hps=True,
        )
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        for j in range(self.J_phi):
            x, _ = self.phi(x)
            x = x[:,:,::2]
        # ReLU ensures the output, which is a smoothed approximation
        # of the modulus, is non-negative
        x = self.relu(x)
        return x
