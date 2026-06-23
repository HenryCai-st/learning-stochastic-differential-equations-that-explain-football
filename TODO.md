1. **Generate training and validation dataset**

- use rho in [0, 20, 40], in 2 regimes
- sigma [x, y]
- beta [a, b]
- noise [0, 0.1]
- T = 5, dt = 0.01/1e-3
- sensitivity
- result should be 5 dataset, 1 with all left, each right one
- each category 10 images, sanity check for training
- use mother seed to generate child seeds for each parameter set
- label rho = 0 fixed, = 20 limit, = 40 chaos
- split rho = [0, 40] and [20] as train validation dataset
- resolution 256x256
- parameters in json

2. **Dataloader**

- rescale to 224x224 for ResNet
- normalize image to [-1,1] using minmax, or use from ResNet (fix mean and std)
- rescale parameters to [-1,1], noise log transform 1st

3. **Model Architecture**

- CNN with frozen/pretrained layers ResNet
- output 4 values array, [sigma, rho, beta, noise]
- loss: l1 loss
- train much to overfit

4. **Training cycle**

- any optimizer params
- output train loss and val loss per epoch to csv
- better if additionally plot and save

5. **Real dataset and training**

- refined distribution of 4 parameters, refer to design notes
- generate 100/500 each using grid mode
- split train test accordingly
- apply on real training
