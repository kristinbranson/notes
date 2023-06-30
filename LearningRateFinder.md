# Learning Rate Finder

I was interested in a general method for setting the initial learning rate. The method used in Pytorch Lighning, FastAI, and Athena is based on [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) by Leslie N. Smith. To search for a good initial learning rate between some small learning rate $\lambda_1$ and some large learning rate $lambda_2$, one starts with learning rate $\lambda_1$ and iteratively increases the learning rate to $\lambda_2$, using either a log or linear schedule. One evaluates the loss at each step on either another batch of the training data or a validation dataset. At small learning rates, the change in loss should be essentially 0. At too large learning rates, the loss should increase or fluctuate a lot. The heuristic is to choose the learning rate for which there is the steepest decrease in loss. Seems like a reasonable idea! 

I used the package [pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder) which hasn't been updated since 2020 but seems to work. I combined two of the ideas from the sample code -- using a log-scheduler and a validation data set:

```
from torch_lr_finder import LRFinder

model = ...
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=500, num_iter=100,val_loader=testloader)

lrs = np.array(lr_finder.history["lr"])
losses = np.array(lr_finder.history["loss"])
fig,ax = plt.subplots(2,1,sharex=True)
lr_finder.plot(ax=ax[0]) # to inspect the loss-learning rate graphQ
ax[0].lines[0].set_marker('o')
dloss = losses[1:]-losses[:-1]
ax[1].plot(lrs[:-1],dloss,'o-')
ax[1].set_xscale('log')
ax[1].set_yscale('symlog')
ax[1].set_ylabel('Change in loss')
min_grad_idx = (np.gradient(losses)).argmin()
learning_rate = lrs[min_grad_idx]
ax[1].plot(lrs[min_grad_idx],dloss[min_grad_idx],'ro')

lr_finder.reset() # to reset the model and optimizer to their initial state
```

Some requirements:
* `trainloader` and `testloader` must output two tensors, the batch inputs and the batch labels:
* `criterion` must take the outputs from the `model`'s forward function and the batch labels and output a single number
That is, the following should work:
```
x,y = next(iter(loader))
loss = criterion(model(x),y)
```

Here's the plot it produced in my test:
![image](https://github.com/kristinbranson/notes/assets/211380/f56ea119-3b05-4138-9fe2-c29437dc27f8)
