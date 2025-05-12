import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torch 


class DiffusionProcess(nn.Module):
    def __init__(
        self,
        noise_schedule, 
        noise_scale, 
        noise_min, 
        noise_max,
        steps, 
        device, 
        keep_num=10
    ):
        super(DiffusionProcess, self).__init__()
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.keep_num = keep_num
        self.Lt_record = th.zeros(steps, keep_num, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)


        #The important parameters for gaussian diffusion
        self.beta_nums = th.tensor(self.betas_num(), dtype=th.float64).to(self.device)
        assert len(self.beta_nums.shape) == 1, "betas must be 1-D"
        assert len(self.beta_nums) == self.steps, "num of betas must equal to diffusion steps"
        assert (self.beta_nums > 0).all() and (self.beta_nums <= 1).all(), "betas out of range"

        self.diffusion_setting()

    def betas_num(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        st_bound = self.noise_scale * self.noise_min
        e_bound = self.noise_scale * self.noise_max
        if self.noise_schedule == "linear":
            return np.linspace(st_bound, e_bound, self.steps, dtype=np.float64)
        else:
            return betas_from_linear_variance(self.steps, np.linspace(st_bound, e_bound, self.steps, dtype=np.float64))

    def diffusion_setting(self):
        alphas = 1.0 - self.beta_nums
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.beta_nums * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.beta_nums * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    

    def caculate_losses(self, model, emb_s, reweight=False):
        batch_size, device = emb_s.size(0), emb_s.device
        ts, pt = self.sample_timesteps(batch_size, device, 'uniform')
        noise = th.randn_like(emb_s)
        emb_t = self.forward_process(emb_s, ts, noise)
        terms = {}
        model_output = model(emb_t, ts)


        assert model_output.shape == emb_s.shape

        mse = mean_flat((emb_s - model_output) ** 2)

        if reweight == True:

            weight = self.SNR(ts - 1) - self.SNR(ts)
            weight = th.where((ts == 0), 1.0, weight)
            loss = mse

        else:
            weight = th.tensor([1.0] * len(model_output)).to(device)

        terms["loss"] = weight * loss
        terms["pred_xstart"] = model_output
        return terms

    def p_sample(self, model, emb_s, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            emb_t = emb_s
        else:
            t = th.tensor([steps - 1] * emb_s.shape[0]).to(emb_s.device)
            emb_t = self.q_sample(emb_s, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * emb_t.shape[0]).to(emb_s.device)
                emb_t = model(emb_t, t)
            return emb_t

        for i in indices:
            t = th.tensor([i] * emb_t.shape[0]).to(emb_s.device)
            out = self.p_mean_variance(model, emb_t, t)
            if sampling_noise:
                noise = th.randn_like(emb_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(emb_t.shape) - 1)))
                )  # no noise when t == 0
                emb_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                emb_t = out["mean"]
        return emb_t
    


    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.keep_num).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            Lt_sqrt = th.sqrt(th.mean(self.Lt_record ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == 'uniform':  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt

        else:
            raise ValueError

    def forward_process(self, emb_s, t, noise=None):
        if noise is None:
            noise = th.randn_like(emb_s)
        assert noise.shape == emb_s.shape
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, emb_s.shape) * emb_s
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, emb_s.shape)
                * noise
        )

    def q_posterior_mean_variance(self, emb_s, emb_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert emb_s.shape == emb_t.shape
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, emb_t.shape) * emb_s
                + self._extract_into_tensor(self.posterior_mean_coef2, t, emb_t.shape) * emb_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, emb_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, emb_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == emb_s.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        pred_xstart = model_output

        model_mean, _, _ = self.q_posterior_mean_variance(emb_s=pred_xstart, emb_t=x, t=t)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, emb_t, t, eps):
        assert emb_t.shape == eps.shape
        return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, emb_t.shape) * emb_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, emb_t.shape) * eps
        )

    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)



def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class SDNet(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(SDNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
