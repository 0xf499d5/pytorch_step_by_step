
# Defines number of epochs
n_epochs = 200  # TODO: why???

losses = []
val_losses = []

for epoch in range(n_epochs):
    # inner loop
    # TRAINING
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    # VALIDATION
    # no gradients in validation!
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)
