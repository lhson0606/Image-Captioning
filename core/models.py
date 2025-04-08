from torch import nn
import torchvision
from config import gconfig


class Encoder(nn.Module):
    """
      Encoder
      We are using a pretrained ResNet-101 model as the encoder.
      :param encoded_image_size: size of the encoded image
      :return a tensor of shape (batch_size, 2048, encoded_image_size, encoded_image_size)
    """
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()

        self.enc_image_size = encoded_image_size

        # Load the pretrained ResNet-101 model
        resnet = torchvision.models.resnet101(pretrained=True)
        # According to https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning?tab=readme-ov-file#dataset
        # We need to remove the last two layers of the ResNet-101 model because we are not doing classification.
        # Todo: I don't know why for now!
        # Remove the last two layers
        modules = list(resnet.children())[:-2]
        # Our convolutional neural network
        self.resnet = nn.Sequential(*modules)
        # Resize the image to a fixed size to allow input images of variable size using adaptive average pooling
        # e.g: 7x7, 15x15, 3x3 will be converted to 14x14
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Fine-tune the model
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # Pass the images through the ResNet-101 model
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        # Resize the image to a fixed size
        out = self.adaptive_pool(out)
        # We follow pytorch convention Batch x Channel x Height x Width (BCHW)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        # Prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        # Todo: Not sure for now!
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: Feature size of encoded images
        :param decoder_dim: Size of decoder's hidden state
        :param attention_dim: Size of attention network's intermediate layer
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Linear layer to transform decoder's hidden state
        self.full_att = nn.Linear(attention_dim, 1)  # Linear layer to compute attention weights
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out:
        :param decoder_hidden:
        :return:
        """
        # Transform the encoder output
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        # Transform the decoder hidden state
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att2 = att2.unsqueeze(1)  # (batch_size, 1, attention_dim)

        # Combine and score the attention weights
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch_size, num_pixels)




