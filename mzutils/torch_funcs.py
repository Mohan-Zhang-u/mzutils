import sys
import torch
import torchvision


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


class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=True, fill=0, padding_mode='constant'):
        """[summary]

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

encode_transform = torchvision.transforms.Compose(
    [
        PadCenterCrop(2048, pad_if_needed=True, fill='white'),
    ]
)
