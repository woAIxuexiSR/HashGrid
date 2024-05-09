import torch
import torch.nn as nn
import numpy as np

# from src.knn_grid import KnnGrid
from hash_grid import HashGrid
from rand_grid import RandGrid
from pyramid_grid import PyramidGrid
# from tcnn_grid import TCNNGrid

from PIL import Image
from tqdm import tqdm


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        # self.grid = HashGrid(input_dim=2)
        # self.grid = RandGrid(input_dim=2)
        # self.grid = KnnGrid(2, input_dim=2)
        self.grid = PyramidGrid(input_dim=2)
        # self.grid = TCNNGrid(input_dim=2)
        self.mlp = nn.Sequential(
            nn.Linear(self.grid.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.grid(x).float()
        x = self.mlp(x)
        return x


if __name__ == "__main__":

    # params
    num_epochs = 1000
    batch_size = 2**14

    # prepare img
    img = Image.open("albert.jpg")
    img = torch.from_numpy(np.array(img) / 255.0).float()

    # prepare model
    model = Model().cuda()

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    model.train()
    tqdm_iter = tqdm(range(num_epochs))
    for i in tqdm_iter:

        x = torch.rand(batch_size, 2)

        coords = x * (torch.tensor(img.shape[:2]) - 1)
        xi = coords.long()
        yi = xi + 1
        xf = coords - xi.float()
        yf = 1 - xf

        p0 = img[xi[:, 0], xi[:, 1]]
        p1 = img[xi[:, 0], yi[:, 1]]
        p2 = img[yi[:, 0], xi[:, 1]]
        p3 = img[yi[:, 0], yi[:, 1]]

        res = (
            p0 * yf[:, 0] * yf[:, 1]
            + p1 * yf[:, 0] * xf[:, 1]
            + p2 * xf[:, 0] * yf[:, 1]
            + p3 * xf[:, 0] * xf[:, 1]
        )

        x, y = x.cuda(), res.reshape(-1, 1).cuda()
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optim.step()
        optim.zero_grad()

        tqdm_iter.set_description(f"loss: {loss.item():.4f}")

    # test
    model.eval()

    res = (1080, 1400)
    x = (np.arange(res[1], dtype=np.float32) + 0.5) / res[1]
    y = (np.arange(res[0], dtype=np.float32) + 0.5) / res[0]
    x, y = np.meshgrid(x, y)

    input = torch.tensor(np.stack([x, y], axis=-1).reshape(-1, 2)).cuda()
    pred = model(input).cpu().detach().numpy().reshape(res)
    pred = pred.transpose(1, 0)

    img = Image.fromarray((pred * 255).astype(np.uint8))
    img.save("out.jpg")
