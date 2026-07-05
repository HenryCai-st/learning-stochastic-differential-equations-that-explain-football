1. What is valid range of parameters? we have different regimes (fixed, limit, chaos) based on choice of rho and/or other parameters, ratios of parameters sometimes have the same pattern.

2. Generated sde images with fixed axis length or scaled, that we only care about patterns not size

3. For dataloader, how do we scale each parameter(min max or z-score, whether log transform for noise scale) and image(black white to [-1,1]) for training

4. Do we label each sde and only train 2/3 regimes instead of all 3? how do we label based on parameters only?

5. do we initialize y0s all from 0 or differently? How do they affect the pattern?