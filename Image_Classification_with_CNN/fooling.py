
from torch.autograd import Variable


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 1, 28, 28)
    - target_y: An integer in the range [0, 9)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and wrap it in a Variable.
    X_fooling = X.clone()
    X_fooling_var = Variable(X_fooling, requires_grad=True)

    learning_rate = 1

    # Train loop
    for i in range(100):

        # Forward.
        scores = model(X_fooling_var)

        # Current max index.
        _, index = scores.data.max(dim=0)

        # bp()
        # # Break if we've fooled the model.
        # if index[0, 0] == target_y:
        #     break

        # Score for the target class.
        target_score = scores[0, target_y]

        # Backward.
        target_score.backward()

        # Gradient for image.
        im_grad = X_fooling_var.grad.data

        # Update our image with normalised gradient.
        X_fooling_var.data += learning_rate * (im_grad / im_grad.norm())

        # Zero our image gradient.
        X_fooling_var.grad.data.zero_()

    return X_fooling



