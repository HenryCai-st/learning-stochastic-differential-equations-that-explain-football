1. **generate train and val dataset**
- use rho in [0, 20, 40], in 2 regimes
- sigma [x,y]
- beta [a,b]
- noise [0, 0.1]
- T = 5, dt = 0.01/1e-3
- sensitivity
- result should be 5 dataset, 1 with all left, each right one
- each category 10 images, sanity chec for training
- use mother seed to generate child seeds for each parameter set
- label rho = 0 fixed, = 20 limit, = 40 chaos
- split rho = [0, 40] and [20] as train val dataset
- resolution 256
- parameters in son

2. **dataloader**
- rescale to 224 for resnet
- normalize image to [-1,1] using minmax, or use from resnet (fix mean and std)
- rescale parameters to [-1,1], noise log transform 1st

3. **model architectute**
- cnn with frozen/pretrained layers resnet
- output 4 values array, [sigma, rho, beta, noise]
- loss l1 loss
- train much to overfit

4. **training cycle**
- any optimizer params
- output train loss and val loss per epoch to csv
- better if additionaly plot and save

1. **real dataset and training**
- refined distribution of 4 parameters, refer cto design notes
- generate 100/500 each using grid mode
- split train test accordingly
- apply on real training