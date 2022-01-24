# JamesGAN
This was my first attempt at creating a deep convolutional GAN to generate images. I used pictures of my face as the domain/target for a few reasons; for one, it is 
quite hard to find a dataset with enough images that are similar (in profile, pose, etc) given how notoriously data-intensive GANs are. Because of this, most GANs done for practice 
involve the same few datasets. On top of that, seeing images of my face that were never taken with a camera before slowly develop made for a far more enjoyable and 
personal experience. 

The results ended up better than I expected, after a ton of tuning and experimenting with different hyperparameters. It took approximately 5000 epochs to end up with 
acceptable results. Even with the high amount of consistency between the images, the model was predictably still extremely unstable. Additionally, extreme care had to be taken throughout 
training to avoid mode collapse into a single output image. I tried many attempts at regularization, with varying success from using the Wasserstein Loss Function, implementing label 
smoothing (allowing labels to deviate slightly from only 1 and 0), and utilizing different nonlinearities. Eventually, I ended up with the following "transformation" gif of 
generated images developing. The below animation shows images generated with the same latent dimensions by models at different points in the training process. 

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

Overall, I thoroughly enjoyed this project, and I hope to experiment with different GAN architectures in the future. I have plans to work on progressively growing
GANs, as well as applications such as image-to-image, image segmentation, and style transfer models. 
