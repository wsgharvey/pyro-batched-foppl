# from batch_pyro.batch_hmc import BatchHMC
import pyro

cond_model = pyro.condition(mnist_model,
                            data={'digit_label': torch.tensor([3, 3])})
hmc = BatchHMC(cond_model, batch_size=2)
print(hmc._adapter._r_dist)
hmc.setup(5, batch_size=2, observed_pixels={666: torch.tensor(0.)})
