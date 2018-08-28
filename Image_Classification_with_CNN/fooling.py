from torch.autograd import Variable
import numpy as np

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 1, 28, 28)
    - target_y: An integer in the range [0, 9)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that tried to classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and wrap it in a Variable.
    X_fooling = X.clone()
    X_fooling_var = Variable(X_fooling, requires_grad=True)

    learning_rate = 1

    # Train loop
    for i in range(1,500):

        # Forward.
        scores = model(X_fooling_var)

        # Current max index.
        index = scores.data.max(dim=1)

        # Break if we've fooled the model.
        if index == target_y:
            print("model fooled at iteration : "+str(i))
            break

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


###############################################################################################################################################################

#One Pixel Attack
def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,], ...]
        pixels = np.split(x, len(x) // 3)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb

    return imgs