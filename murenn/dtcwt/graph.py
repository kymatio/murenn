import torch
import murenn


class MURENN_GRAPH(torch.nn.Module):
    def __init__(
        self,
        J=8,
        Q=4,
        T=16,
        padding_mode="symmetric",
    ):
        super().__init__()
        self.J = J
        self.Q = Q
        self.T = T
        self.padding_mode = padding_mode
        self.dtcwt = murenn.DTCWT(
            J=self.J,
            padding_mode=self.padding_mode,
        )


    def forward(self, x):
        _, bps = self.dtcwt(x)
        ys = []
        C = bps[0].shape[1]

        for j in range(self.J):
            down = murenn.DTCWT(
                J=self.J-j,
                padding_mode=self.padding_mode,
                skip_hps=True,
            )

            conv1d = torch.nn.Conv1d(
                in_channels=C,
                out_channels=self.Q,
                kernel_size=self.T,
                bias=False,
                padding="same",
            )
            torch.nn.init.normal_(conv1d.weight) #std?

            y_j, _ = down(conv1d(torch.abs(bps[j])))
            ys.append(y_j)

        ys = torch.stack(ys, dim=2) #B*C*J*N

        return ys