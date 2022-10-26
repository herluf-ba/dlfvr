# TODO: these functions might end up being reusable. 
# Find a way to move them somewhere else if necessray. 

def loss_batch(model, loss_func, xb, yb, opt=None):
    print(xb.shape, yb.shape)
    exit()
    joemama = model.forward(xb)
    print(joemama.shape)
    exit()
    loss = loss_func(joemama, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

    # Stolen from the above mentioned tutorial - TODO: store somewhere 
    # as a general function for training

def fit(model, epochs, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

