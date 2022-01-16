import sys
import numpy as np
import torch
import torchvision
from torch.cuda.amp import custom_bwd, custom_fwd


def model_params(model: torch.nn.Module) -> int:
    """
    an easier solution as compare to https://github.com/amarczew/pytorch_model_summary
    :param model:
    :return: the number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv2d_select_correct_element(c_in: int or tuple, loc):
    if isinstance(c_in, int):
        return c_in
    else:
        return c_in[loc]


def conv2d_output_single_shape(h_in: int, kernel_size: int, stride: int = 1, padding: int or tuple = 0,
                               dilation: int = 1):
    """
    compute the output shape of a single height or width after convolve, as the formula here states:
    https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html
    :param h_in:
    :param padding:
    :param dilation:
    :param kernel_size:
    :param stride:
    :return:
    """
    if isinstance(padding, int):
        return int((h_in + 2. * padding - dilation * (kernel_size - 1.) - 1.) / stride + 1.)
    else:
        return int((h_in + sum(padding) - dilation * (kernel_size - 1.) - 1.) / stride + 1.)


def conv2d_output_shape(out_channels: int, H_in: int, W_in: int, kernel_size: int or tuple, stride: int or tuple = 1,
                        padding: int or tuple = 0, dilation: int or tuple = 1):
    if isinstance(padding, int):
        H = conv2d_output_single_shape(H_in, conv2d_select_correct_element(kernel_size, 0),
                                       conv2d_select_correct_element(stride, 0), padding,
                                       conv2d_select_correct_element(dilation, 0))
        W = conv2d_output_single_shape(W_in, conv2d_select_correct_element(kernel_size, 1),
                                       conv2d_select_correct_element(stride, 1), padding,
                                       conv2d_select_correct_element(dilation, 1))
        return out_channels, H, W
    else:
        H = conv2d_output_single_shape(H_in, conv2d_select_correct_element(kernel_size, 0),
                                       conv2d_select_correct_element(stride, 0), padding[0:2],
                                       conv2d_select_correct_element(dilation, 0))
        W = conv2d_output_single_shape(W_in, conv2d_select_correct_element(kernel_size, 1),
                                       conv2d_select_correct_element(stride, 1), padding[2:4],
                                       conv2d_select_correct_element(dilation, 1))
        return out_channels, H, W


def check_tensor_occupied_memory(t):
    """
    this is a reminder function.
    """
    print(sys.getsizeof(t.storage()))


class LabelSmoothingLoss(torch.nn.Module):
    # https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def differentiable_clamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    >>> x=torch.from_numpy(np.array(([0])).astype(np.float32)).requires_grad_()
    >>> a=torch.from_numpy(np.array(([1])).astype(np.float32))
    >>> a.require_grad=False
    >>> (x - a).pow(2).backward()
    >>> x.grad
    tensor([-2.])
    >>> x=torch.from_numpy(np.array(([0])).astype(np.float32)).requires_grad_()
    >>> a=torch.from_numpy(np.array(([1])).astype(np.float32))
    >>> a.require_grad=False
    >>> (x.clamp(min=0.5) - a).pow(2).backward()
    >>> x.grad
    tensor([0.])
    >>> x=torch.from_numpy(np.array(([0])).astype(np.float32)).requires_grad_()
    >>> a=torch.from_numpy(np.array(([1])).astype(np.float32))
    >>> a.require_grad=False
    >>> (differentiable_clamp(x, min=-0.5, max=2) - a).pow(2).backward()
    >>> x.grad
    tensor([-2.])
    >>> x=torch.from_numpy(np.array(([0])).astype(np.float32)).requires_grad_()
    >>> a=torch.from_numpy(np.array(([1])).astype(np.float32))
    >>> a.require_grad=False
    >>> (differentiable_clamp(x, min=0.5, max=2) - a).pow(2).backward()
    >>> x.grad
    tensor([-1.])
    """
    return DifferentiableClamp.apply(input, min, max)


class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=True, fill=0, padding_mode='constant'):
        """useage:
        encode_transform = torchvision.transforms.Compose(
            [
                PadCenterCrop(2048, pad_if_needed=True, fill='white'),
            ]
        )
        a=Image.open('1.jpg')
        encode_transform(a)
        
        Args:
            size ([type]): same as torchvision.transforms.CenterCrop's size
            pad_if_needed (bool, optional): Defaults to True. if True, behave the same as CenterCrop.
            fill ([type], optional): same as torchvision.transforms.Pad's fill. Defaults to 0.
            padding_mode ([type], optional): same as torchvision.transforms.Pad's padding_mode. Defaults to 0.
        """
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img):
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = torchvision.transforms.functional.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = torchvision.transforms.functional.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return torchvision.transforms.functional.center_crop(img, self.size)

class ImageExperimentProcessor:
    def __init__(self, image_size, pad_center_crop=False, resize=True, to_tensor=True, normalize=True, normalize_mean=np.array([0.485, 0.456, 0.406]), normalize_std=np.array([0.229, 0.224, 0.225])):
        self.image_size = image_size
        self.pad_center_crop = pad_center_crop
        self.resize = resize
        self.to_tensor = to_tensor
        self.normalize = normalize
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
    def preprocess_transform_one_img(self, pil_img):
        """
        img_tensor is one image tensor whose len of shape is 3.
        order:
        PadCenterCrop,
        Resize,
        ToTensor
        Normalize,
        """
        w, h = pil_img.size
        transform_list = []
        if self.pad_center_crop:
            max_len = max(w, h)
            transform_list.append(PadCenterCrop(max_len, pad_if_needed=True, fill='white'))
        if self.resize:
            transform_list.append(torchvision.transforms.Resize(size=(self.image_size, self.image_size))) # h, w
        if self.to_tensor:
            transform_list.append(torchvision.transforms.ToTensor()) # input image is scaled to [0.0, 1.0]
        if self.normalize:
            transform_list.append(torchvision.transforms.Normalize(self.normalize_mean.tolist(), self.normalize_std.tolist()))
        encode_transform = torchvision.transforms.Compose(transform_list)
        img_tensor = encode_transform(pil_img)
        # pil_img.save('pil_img.jpg')
        return img_tensor
    
    def preprocess_transform_imgs(self, pil_imgs, device='cpu'):
        """
        pil_imgs: list of PIL.Image
        """
        img_sizes = [img.size for img in pil_imgs] # [(w, h)]
        img_tensors = [self.preprocess_transform_one_img(pil_img) for pil_img in pil_imgs]
        img_tensors = torch.stack(img_tensors).to(device)
        return img_tensors, img_sizes
    
    def postprocess_transform_one_tensor(self, img_tensor, h=256, w=256):
        """
        img_tensor is one image tensor whose len of shape is 3.
        Normalize,
        ToPILImage,
        Resize,
        PadCenterCrop,
        """
        transform_list = []
        if self.normalize:
            transform_list.append(torchvision.transforms.Normalize((-self.normalize_mean / self.normalize_std).tolist(), (1.0 / self.normalize_std).tolist()))
        if self.to_tensor:
            transform_list.append(torchvision.transforms.ToPILImage())
        if self.pad_center_crop:
            if self.resize:
                max_len = max(w, h)
                transform_list.append(torchvision.transforms.Resize(size=(max_len, max_len)))
            transform_list.append(torchvision.transforms.CenterCrop(size=(h, w)))
            
        else:
            if self.resize:
                transform_list.append(torchvision.transforms.Resize(size=(h, w)))
        decode_transform = torchvision.transforms.Compose(transform_list)
        pil_img = decode_transform(img_tensor)
        # pil_img.save('pil_img.jpg')
        return pil_img
    
    def postprocess_transform_tensors(self, img_tensors, img_sizes):
        """
        img_tensors: list of torch.Tensor or torch.Tensor of torch.Size([c, self.image_height, self.image_width])
            or a torch.Tensor of torch.Size([img.shape[0], c, self.image_height, self.image_width])
        img_sizes: list of (w, h)
        """
        assert len(img_tensors) == len(img_sizes)
        converted_imgs = []
        for i in range(len(img_tensors)):
            converted_img = self.postprocess_transform_one_tensor(img_tensors[i], w=img_sizes[i][0], h=img_sizes[i][1])
            converted_imgs.append(converted_img)
        return converted_imgs
