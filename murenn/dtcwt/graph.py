import torch
import murenn


class MuReNNDirect(torch.nn.Module):
    def __init__(self, *, J, Q, T, in_channels, padding_mode="symmetric"):
        super().__init__()
        self.Q = Q
        self.ch = in_channels
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
        assert self.ch == x.shape[1]
        _, bps = self.dtcwt(x)
        ys = []

        for j in range(self.dtcwt.J):
            x_j = torch.abs(bps[j])
            x_j = self.conv1d[j](x_j)
            y_j, _ = self.down[j](x_j)

            B, _, N = y_j.shape
            y_j = y_j.view(B, self.ch, self.Q, N)
            ys.append(y_j)

        y = torch.stack(ys, dim=3) #shape: (B, ch, Q, J, N/(2**J))
        return y