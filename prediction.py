import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt


# Utils
def layer_init_xavier(layer, bias=True):
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.constant_(layer.bias.data, 0)
    return layer

def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x

# Network base
class FCBody(nn.Module):
    def __init__(self, device, input_dim, hidden_units, activation=functional.relu):
        super().__init__()
        self.to(device)
        self.device = device
        dims = (input_dim,) + hidden_units
        self.layers = nn.ModuleList([layer_init_xavier(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.activation = activation
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class FCNetwork(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, head_activation=lambda x:x):
        super().__init__()
        body = FCBody(device, input_units, hidden_units=tuple(hidden_units))
        self.body = body
        self.fc_head = layer_init_xavier(nn.Linear(body.feature_dim, output_units, bias=True), bias=True)
        self.device = device
        self.head_activation = head_activation
        self.to(device)

    def forward(self, x):
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        y = self.body(x)
        y = self.fc_head(y)
        y = self.head_activation(y)
        return y


# Models
class Predictor(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units,
                 head_activation=lambda x: x):
        super().__init__()
        self.predictor = FCNetwork(device, input_units, hidden_units, output_units,
                                   head_activation=head_activation)

    def forward(self, x):
        prediction = self.predictor(x)
        return prediction

    def eval(self, x):
        with torch.no_grad():
            prediction = self.predictor(x)
        return prediction


class VAE(nn.Module):
    def __init__(self, device, input_units, hidden_units, rep_units, head_activation=lambda x: x):
        super().__init__()
        self.encoder = FCNetwork(device, input_units, hidden_units, rep_units,
                                 head_activation=head_activation)
        self.decoder = FCNetwork(device, rep_units, hidden_units[::-1], input_units,
                                 head_activation=lambda x: x) # Head activation has to be identity

    def forward(self, x):
        rep = self.encoder(x)
        out = self.decoder(rep)
        return rep, out

    def eval(self, x):
        with torch.no_grad():
            rep = self.encoder(x)
        return rep

class TwoHeadedPredictor(nn.Module):
    def __init__(self, device, input_units, hidden_units, rep_units, output_units,
                 rep_head_activation=lambda x: x, predictor_head_activation=lambda x: x):
        super().__init__()
        self.encoder = FCNetwork(device, input_units, hidden_units, rep_units,
                                 head_activation=rep_head_activation)
        self.decoder = FCNetwork(device, rep_units, hidden_units[::-1], input_units,
                                 head_activation=lambda x: x)  # Head activation has to be identity
        self.predictor = FCNetwork(device, rep_units, hidden_units[::-1], output_units,
                                 head_activation=predictor_head_activation)

    def forward(self, x):
        rep = self.encoder(x)
        vae_out = self.decoder(rep)
        prediction = self.predictor(rep)
        return rep, vae_out, prediction

    def eval(self, x):
        with torch.no_grad():
            rep = self.encoder(x)
            prediction = self.predictor(rep)
        return rep, prediction



# Training code
class SupervisedLrSchduler:
    def __init__(self, eps=1e-5, threshold=3):
        self.eps = eps
        self.threshold = threshold
        self.training_loss = []
        return

    def __call__(self, optimizer, loss):
        self.training_loss.append(loss)
        if len(self.training_loss) < self.threshold:
            return
        if self.__check_converge():
            print("Reducing learning rate")
            self.training_loss = []
            for g in optimizer.param_groups: # should be an in-position change
                g['lr'] = g['lr'] / 2.

    def __check_converge(self):
        length = len(self.training_loss)
        for i in range(1, self.threshold):
            # print(length, length-i, length-i-1)
            if self.training_loss[length-i-1] - self.training_loss[length-i] >= self.eps:
                return False
        return True



def init_optimizer(name, param, lr):
    if name == "RMSprop":
        return torch.optim.RMSprop(param, lr)
    elif name == 'Adam':
        return torch.optim.Adam(param, lr)
    elif name == "SGD":
        return torch.optim.SGD(param, lr)
    else:
        raise NotImplementedError

def two_step_trainer(inputs_all, targets_all, epochs, batch_size,
                     vae_lr, vae_loss_fn, vae_optimizer_name,
                     predictor_lr, predictor_loss_fn, predictor_optimizer_name,
             device, input_units, hidden_units, rep_units, output_units, head_activation=lambda x: x,
             test_prop=0.1, seed=1024):

    rng = np.random.RandomState(seed)
    test_idx = rng.choice(len(inputs_all), size=int(len(inputs_all)*test_prop), replace=False)
    train_idx = list(set(np.arange(len(inputs_all))) - set(test_idx))

    training_inputs = tensor(inputs_all[train_idx], device).to(device)
    training_targets = tensor(targets_all[train_idx], device).to(device)
    test_in_tensor = tensor(inputs_all[test_idx], device).to(device)
    test_tar_tensor = tensor(targets_all[test_idx], device).to(device)

    dataset = data_utils.TensorDataset(training_inputs, training_targets)
    train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae_model = VAE(device, input_units, hidden_units, rep_units)
    vae_optimizer = init_optimizer(vae_optimizer_name, list(vae_model.parameters()), vae_lr)
    vae_lr_schdl = SupervisedLrSchduler()

    for ei in range(1, epochs+1):
        avg_loss = []
        for batch_idx, (inputs, _) in enumerate(train_loader):
            _, pred = vae_model(inputs)

            vae_optimizer.zero_grad()
            loss = vae_loss_fn(pred, inputs)
            loss.backward()
            vae_optimizer.step()

            avg_loss.append(loss.detach().cpu().numpy())
        al = np.mean(np.array(avg_loss))
        print("VAE training | Epoch [{}/{}], Training loss: {:.4f}".format(ei, epochs, al))
        vae_lr_schdl(vae_optimizer, al)

    predictor = Predictor(device, rep_units, hidden_units, output_units)
    predictor_optimizer = init_optimizer(predictor_optimizer_name, list(predictor.parameters()), predictor_lr)
    predictor_lr_schdl = SupervisedLrSchduler()
    for ei in range(1, epochs+1):
        avg_loss = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            rep = vae_model.eval(inputs)
            pred = predictor(rep)

            predictor_optimizer.zero_grad()
            loss = predictor_loss_fn(pred, targets)
            loss.backward()
            predictor_optimizer.step()

            avg_loss.append(loss.detach().cpu().numpy())
        al = np.mean(np.array(avg_loss))
        print("Predictor training | Epoch [{}/{}], Training loss: {:.4f}".format(ei, epochs, al))
        predictor_lr_schdl(predictor_optimizer, al)

    rep = vae_model.eval(test_in_tensor)
    pred = predictor.eval(rep)
    loss = predictor_loss_fn(pred, test_tar_tensor)
    print("Evaluation | Test loss: {:.4f}".format(loss))

    rep_all = vae_model.eval(tensor(inputs_all, device).to(device))
    pred_all = predictor.eval(rep_all)
    return (test_in_tensor.detach().cpu().numpy(), test_tar_tensor.detach().cpu().numpy(), pred.detach().cpu().numpy(),
            pred_all.detach().cpu().numpy())

def multiheaded_trainer(inputs_all, targets_all, epochs, batch_size,
                        lr, loss_fn, optimizer_name, auxiliary_weight,
                        device, input_units, hidden_units, rep_units, output_units,
                        rep_head_activation=lambda x: x,
                        predictor_head_activation=lambda x: x,
                        test_prop=0.1, seed=1024):

    rng = np.random.RandomState(seed)
    test_idx = rng.choice(len(inputs_all), size=int(len(inputs_all)*test_prop), replace=False)
    train_idx = list(set(np.arange(len(inputs_all))) - set(test_idx))

    training_inputs = tensor(inputs_all[train_idx], device).to(device)
    training_targets = tensor(targets_all[train_idx], device).to(device)
    test_in_tensor = tensor(inputs_all[test_idx], device).to(device)
    test_tar_tensor = tensor(targets_all[test_idx], device).to(device)

    dataset = data_utils.TensorDataset(training_inputs, training_targets)
    train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TwoHeadedPredictor(device, input_units, hidden_units, rep_units, output_units,
                               rep_head_activation=rep_head_activation, predictor_head_activation=predictor_head_activation)
    optimizer = init_optimizer(optimizer_name, list(model.parameters()), lr)
    lr_schdl = SupervisedLrSchduler()

    for ei in range(1, epochs+1):
        avg_loss = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            _, vae_out, pred = model(inputs)

            optimizer.zero_grad()
            loss_main = loss_fn(pred, targets)
            loss_aux = loss_fn(vae_out, inputs)
            loss = loss_main + auxiliary_weight * loss_aux
            loss.backward()
            optimizer.step()

            avg_loss.append(loss.detach().cpu().numpy())
        al = np.mean(np.array(avg_loss))
        print("Training | Epoch [{}/{}], Training loss: {:.4f}".format(ei, epochs, al))
        lr_schdl(optimizer, al)

    _, pred = model.eval(test_in_tensor)
    loss = loss_fn(pred, test_tar_tensor)
    print("Evaluation | Test loss: {:.4f}".format(loss))

    _, pred_all = model.eval(tensor(inputs_all, device).to(device))
    return (test_in_tensor.detach().cpu().numpy(), test_tar_tensor.detach().cpu().numpy(), pred.detach().cpu().numpy(),
            pred_all.detach().cpu().numpy())

def baseline_trainer():
    pass

# For test only
def test_two_step():
    lr = 0.01
    vae_loss_fn = nn.MSELoss()
    predictor_loss_fn = nn.MSELoss()

    x = np.arange(0, 5 * np.pi, 0.01)
    y = np.sin(x)
    x = x / x.max()
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    print("Dataset size", len(x))

    print("-------------- Test two steps training -------------- ")
    ts_test_in, ts_test_tar, ts_test_pred, ts_pred_all = two_step_trainer(
                     x, y,
                     epochs=50,
                     batch_size=16,
                     vae_lr=lr,
                     vae_loss_fn=vae_loss_fn,
                     vae_optimizer_name="Adam",
                     predictor_lr=lr,
                     predictor_loss_fn=predictor_loss_fn,
                     predictor_optimizer_name="Adam",
                     device="cpu",
                     input_units=x.shape[1],
                     hidden_units=(32, 32),
                     rep_units=4,
                     output_units=y.shape[1],
                     head_activation=lambda x: x,
                     test_prop=0.1,
                     seed=1024
                     )

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    x = x.reshape(-1)
    y = y.reshape(-1)
    ts_pred_all = ts_pred_all.reshape(-1)
    ts_test_in, ts_test_pred = ts_test_in.reshape(-1), ts_test_pred.reshape(-1)
    ax.plot(x, y, label="Ground Truth", color="grey")
    ax.plot(x, ts_pred_all, '--', label="Two step all predictions", color="C1")
    ax.scatter(ts_test_in, ts_test_pred, label="Two step tests", color="C1", s=5)
    plt.legend()
    plt.show()

def test_multi_head():
    lr = 0.01

    x = np.arange(0, 5 * np.pi, 0.01)
    y = np.sin(x)
    x = x / x.max()
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    print("Dataset size", len(x))


    print("-------------- Test multi-head training -------------- ")
    loss_fn = nn.MSELoss()
    mh_test_in, mh_test_tar, mh_test_pred, mh_pred_all = multiheaded_trainer(
                        x, y,
                        epochs=100,
                        batch_size=16,
                        lr=lr,
                        loss_fn=loss_fn,
                        optimizer_name="Adam",
                        auxiliary_weight=1,
                        device="cpu",
                        input_units=x.shape[1],
                        hidden_units=(32, 32),
                        rep_units=4,
                        output_units=y.shape[1],
                        rep_head_activation=lambda x: x,
                        predictor_head_activation=lambda x: x,
                        test_prop=0.1,
                        seed=1024
                        )

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    x = x.reshape(-1)
    y = y.reshape(-1)
    mh_pred_all = mh_pred_all.reshape(-1)
    mh_test_in, mh_test_pred = mh_test_in.reshape(-1), mh_test_pred.reshape(-1)
    ax.plot(x, y, label="Ground Truth", color="grey")
    ax.plot(x, mh_pred_all, '--', label="Multi-head all predictions", color="C0")
    ax.scatter(mh_test_in, mh_test_pred, label="Multi-head tests", color="C0", s=5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(1024)
    torch.manual_seed(1024)

    test_two_step()
    test_multi_head()