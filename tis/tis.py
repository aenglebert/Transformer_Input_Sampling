import torch
import torch.nn.functional as F
from torchvision.models import VisionTransformer as VisionVIT
from timm.models.vision_transformer import VisionTransformer as TimmVIT

from fast_pytorch_kmeans import KMeans

from tqdm import tqdm

import math


class TIS:
    def __init__(self,
                 model,
                 n_masks=1024,
                 batch_size=128,
                 tokens_ratio=0.5,
                 normalise=True,
                 verbose=True,
                 ablation_study=False,
                 ):
        """
        Create a TIS class to compute saliency maps for a vision transformer
        :param model: The ViT model to explain
        :param n_masks: The number of masks used to generate the saliency maps
        :param batch_size: Batch size for the computation of the masks scores
        :param tokens_ratio: Ratio of tokens to keep in the masking process
        :param normalise: Bool, normalise the saliency map between [0,1]
        :param verbose: Bool, print information during the computation
        :param ablation_study: Bool, if True, use ablation study mode, implying that the perturbation is done on the
        input image instead of the encoded tokens
        """

        # Check that model is a ViT
        assert isinstance(model, VisionVIT) or isinstance(model, TimmVIT), "Transformer architecture not recognised"

        # Save model
        self.model = model

        # Set parameters
        self.batch_size = batch_size
        self.n_masks = n_masks
        self.normalise = normalise
        self.verbose = verbose
        self.ablation_study = ablation_study

        if isinstance(tokens_ratio, float):
            tokens_ratio = [tokens_ratio]
        self.tokens_ratio = tokens_ratio

        # Initialize working variables
        self.encoder_activations = []
        self.encoder_hook_list = []
        self.cur_mask_indices = None

    def __call__(self, x, class_idx=None):
        """
        The main function called to produce a saliency map
        :param x: tensor (3,  input image for which the map will be generated
        :param class_idx: optional, index of the class to explore
        If not specified, the class predicted by the model will be used
        :return: saliency map, tensor of shape (token_h, token_w)
        """

        # Check that we get only one image
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"

        # Unsqueeze to get 4 dimensions if needed
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        with torch.no_grad():
            # First forward pass to recorde encoder activations
            predicted_class, encoder_activations = self.get_encoder_activations(x)

            # Define the class to explain. If not explicit, use the class predicted by the model
            if class_idx is None:
                class_idx = predicted_class
                if self.verbose:
                    print("class idx", class_idx)

            # Generate the masks
            raw_masks = self.generate_raw_masks(encoder_activations)
            mask_list, mask_indices_list = self.generate_binary_masks(raw_masks)

            # Generate the saliency map for image x and class_idx
            scores = self.generate_scores(x, class_idx, mask_indices_list)

            saliency_map = self.generate_saliency(x, scores, mask_list)

            return saliency_map

    def get_encoder_activations(self, x):
        """
        Retrieve the encoder activations for a given image x
        :param x: image as a tensor
        :return: tuple of predicted_class (int), encoder_activations (tensor)
        """
        # Reset activations and hooks lists
        self.encoder_activations = []
        self.encoder_hook_list = []

        # Define the encoder hook function to retrieve the activations
        def encoder_hook_fn(_, __, output):
            # Store activations into the encoder_activations list
            self.encoder_activations.append(output.detach())

        if isinstance(self.model, VisionVIT):
            layers = self.model.encoder.layers
        elif isinstance(self.model, TimmVIT):
            layers = self.model.blocks
        else:
            print("Model not recognised")
            exit(1)

        # Attach a forward hook to each transformer block
        for layer in layers:
            self.encoder_hook_list.append(layer.register_forward_hook(encoder_hook_fn))

        # Forward pass: get the predicted class and activations are retrieved using the hooks
        predicted_class = torch.argmax(self.model(x))

        # Concatenate the list of activations into a single tensor
        self.encoder_activations = torch.cat(self.encoder_activations, dim=-1)

        # Remove hooks
        for hook in self.encoder_hook_list:
            hook.remove()
        self.encoder_hook_list = []

        return predicted_class, self.encoder_activations

    def generate_raw_masks(self, encoder_activations):
        """
        Generate the masks based on the activations
        :param encoder_activations: tensor of activations
        :return: list of raw masks (list of tensors)
        """
        # Squeeze to shape (n_tokens+1, n_activations)
        encoder_activations = encoder_activations.squeeze(0)

        # remove CLS token and transpose, shape (n_activations, n_tokens)
        encoder_activations = encoder_activations[1:].T

        # Create clusters with kmeans
        kmeans = KMeans(n_clusters=self.n_masks, mode='euclidean', verbose=self.verbose)
        kmeans.fit(encoder_activations)

        # Use kmeans centroids as basis for masks
        raw_masks = kmeans.centroids

        return raw_masks

    def generate_binary_masks(self, raw_masks):
        """
        Generate binary masks based on the raw masks
        :param raw_masks: list of raw masks
        :return: tuple (mask_list, mask_indices_list)
        mask_list is a list of masks (list of tensors)
        mask_indices_list is a list of indices for each mask (list of tensors)
        """

        # Initialise lists for the masks
        mask_indices_list = []
        mask_list = []

        for ratio in self.tokens_ratio:
            for raw_mask in raw_masks:
                # Computer the number of tokens to keep based on the ratio
                n_tokens = int(ratio * raw_mask.flatten().shape[0])

                # Compute the indexes of the n_tokens with the highest values in the raw mask
                mask_indices = raw_mask.topk(n_tokens)[1]

                # Create binary mask
                bin_mask = torch.zeros_like(raw_mask)
                bin_mask[mask_indices] = 1

                # Append current mask to lists
                mask_indices_list.append(mask_indices)
                mask_list.append(bin_mask)

        return mask_list, mask_indices_list

    def mask_input(self, x, baseline="random"):
        """
        Mask the input image x based on the tokens indices in self.cur_mask_indices
        :param x: image tensor
        :param baseline: baseline value for the masked pixels
        :return: masked image tensor
        """
        # Get patch size from the model
        if isinstance(self.model, VisionVIT):
            patch_size = self.model.conv_proj.kernel_size
        elif isinstance(self.model, TimmVIT):
            patch_size = self.model.patch_embed.proj.kernel_size

        # compute number of patches in height and width
        n_patches_h = x.shape[2] // patch_size[0]
        n_patches_w = x.shape[3] // patch_size[1]

        # Create a mask in the shape of the tokens (1D)
        mask_1d = torch.zeros((n_patches_h * n_patches_w)).to(x.device)

        # Create a list of masked images
        masked_images = []

        for indices in self.cur_mask_indices:
            # Set the mask to 0 for all tokens
            mask_1d[:] = 0
            # Set the mask to 1 for the current tokens indices
            mask_1d[indices] = 1

            # Reshape the mask in 2d
            mask = mask_1d.reshape((1, 1, n_patches_h, n_patches_w))

            # Upsample the mask to the size of the image
            mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='nearest')

            if baseline == "random":
                baseline_tensor = torch.rand_like(x)
            elif baseline == "zero":
                baseline_tensor = torch.zeros_like(x)
            else:
                print("Baseline not recognised")
                exit(1)

            # Apply the mask to the image
            masked_images.append(x * mask + baseline_tensor * (1 - mask))

        # Create a batch of masked images
        x = torch.cat(masked_images, dim=0)

        return x

    def generate_scores(self, x, class_idx, mask_indices_list):
        """
        Generate the masks scores
        :param x: Image to produce the saliency map
        :param class_idx: Class to explore for the saliency map
        :param mask_list: List of masks (list of tensors)
        :param mask_indices_list: List of masks indices (list of tensors)
        :return: Score tensor
        """
        # initialise the list of scores of the masks
        scores = []

        # Reset self.cur_mask_indices
        self.cur_mask_indices = None

        # Define the hook to sample tokens based on the current masks
        # It is designed to receive only one set of tokens as input (batch of size 1) and output a batch with the same
        # size as the number of set of masks indices in the list self.cur_mask_indices
        def tokens_sampling_hook_fn(_, __, output):
            # Perform sampling only if a mask is set
            if self.cur_mask_indices is not None:
                # Separate CLS from other tokens
                cls = output[:, 0].unsqueeze(1)
                tokens = output[:, 1:]

                # List of sampled tokens in the batch
                sampled_tokens = []

                # Iterate over the masks of the batch
                for indices in self.cur_mask_indices:
                    # Sample the tokens based on current mask indices
                    cur_tokens = tokens[:, indices]
                    # Add again CLS token
                    cur_tokens = torch.cat([cls, cur_tokens], dim=1)
                    # Add to the list of sampled tokens
                    sampled_tokens.append(cur_tokens)

                # Concatenate the list into a tensor
                sampled_tokens = torch.cat(sampled_tokens)

                return sampled_tokens
                # return torch.cat([cls, sampled_tokens], dim=1)


        if not self.ablation_study:
            # Register the sampling hook at the beginning of the encoder, after the positional embedding
            if isinstance(self.model, VisionVIT):
                tokens_sampling_hook = self.model.encoder.dropout.register_forward_hook(tokens_sampling_hook_fn)
            elif isinstance(self.model, TimmVIT):
                tokens_sampling_hook = self.model.pos_drop.register_forward_hook(tokens_sampling_hook_fn)
            else:
                print("Model not recognised")
                exit(1)

        # Compute scores by batch
        for idx in tqdm(range(math.ceil(len(mask_indices_list) / self.batch_size)), disable=(not self.verbose)):
            # Select the masks attributed to the current batch
            selection_slice = slice(idx * self.batch_size, min((idx + 1) * self.batch_size, len(mask_indices_list)))
            self.cur_mask_indices = mask_indices_list[selection_slice]

            if self.ablation_study:
                # Mask the input
                result = self.model(self.mask_input(x)).detach()
            else:
                # Forward pass with tokens sampling performed by the hook
                result = self.model(x).detach()

            # Get the softmax result for the explored class
            result = torch.softmax(result, dim=1)
            score = result[:, class_idx]

            # Append the scores of the masks in the batch to the list of all scores
            scores.append(score)

        # Remove sampling hook
        if not self.ablation_study:
            tokens_sampling_hook.remove()
        self.cur_mask_indices = None

        # Concatenate all the scores into a tensor
        scores = torch.cat(scores)

        return scores

    def generate_saliency(self, x, scores, mask_list):

        # Stack masks into a tensor
        masks = torch.vstack(mask_list).T

        # Sum the masks weighted by their scores to produce a raw saliency
        scored_masks = scores * masks
        raw_saliency = scored_masks.sum(-1)

        # Compute tokens coverage bias
        coverage_bias = masks.sum(-1)

        if isinstance(self.model, VisionVIT):
            patch_size = (self.model.patch_size, self.model.patch_size)
        elif isinstance(self.model, TimmVIT):
            patch_size = self.model.patch_embed.patch_size
        else:
            print("Model not recognised")
            exit(1)

        # Compute the saliency map height and width
        h = x.shape[-2] // patch_size[0]
        w = x.shape[-1] // patch_size[1]

        # Correct the saliency for coverage bias and reshape and reshape in 2D
        saliency = raw_saliency / coverage_bias
        saliency = saliency.reshape(h, w)

        # Normalise between [0,1] if self.normalise is True
        if self.normalise:
            saliency = saliency - saliency.min()
            saliency = saliency/saliency.max()

        return saliency
