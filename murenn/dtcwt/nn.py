import torch
import murenn


class MuReNNDirect(torch.nn.Module):
    """
    Args:
        J (int): Number of levels of DTCWT decomposition.
        Q (int): Number of Conv1D filters at each level.
        in_channels (int): Number of channels in the input signal.
        padding_mode (str): One of 'symmetric' (default), 'zeros', 'replicate',
            and 'circular'. Padding scheme for the DTCWT decomposition.
    """
    def __init__(self, *, J, Q, T, in_channels, padding_mode="symmetric"):
        super().__init__()
        self.Q = Q
        self.C = in_channels
        self.down = []
        self.conv1d = []
        self.dtcwt = murenn.DTCWT(
            J=J,
            padding_mode=padding_mode,
        )

        for j in range(J):
            down = murenn.DTCWT(
                J=J-j,
                padding_mode=padding_mode,
                skip_hps=True,
            )
            self.down.append(down)

            conv1d = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=Q*in_channels,
                kernel_size=T,
                bias=False,
                groups=in_channels,
                padding="same",
            )
            torch.nn.init.normal_(conv1d.weight)
            self.conv1d.append(conv1d)


    def forward(self, x):
        """
        Args:
            x (PyTorch tensor): Input data. Should be a tensor of shape
                `(B, C, T)` where B is the batch size, C is the number of
                channels and T is the number of time samples.
                Note that T must be a multiple of 2**J, where J is the number
                of wavelet scales (see documentation of MuReNNDirect constructor).

        Returns:
            y (PyTorch tensor): A tensor of shape `(B, C, Q, J, T/(2**J))`
        """
        assert self.C == x.shape[1]
        _, bps = self.dtcwt(x)
        ys = []

        for j in range(self.dtcwt.J):
            Wx_r = self.conv1d[j](bps[j].real)
            Wx_i = self.conv1d[j](bps[j].imag)
            Ux = Wx_r ** 2 + Wx_i ** 2
            y_j, _ = self.down[j](Ux)

            B, _, N = y_j.shape
            # reshape from (B, C*Q, N) to (B, C, Q, N)
            y_j = y_j.view(B, self.C, self.Q, N)
            ys.append(y_j)

        y = torch.stack(ys, dim=3)
        return y