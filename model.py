import numpy
import torch
import torchvision

class ResNet50(torch.nn.Module):
    def __init__(self, num_classes=2, activation=torch.sigmoid, pretrained=True):
        super(ResNet50, self).__init__()
        self._num_classes = num_classes

        resnet_model = torchvision.models.resnet.resnet50(pretrained=pretrained)
        self._features = torch.nn.Sequential(*list(resnet_model.children())[:-1])
        # self._pred = torch.nn.Conv2d(256, 2, kernel_size=[1, 1])

        if isinstance(num_classes, (list, set)):
            self._fc = torch.nn.ModuleList([torch.nn.Linear(2048, x) for x in num_classes])
        else:
            self._fc = torch.nn.Linear(2048, num_classes)

        self._activation = activation

    def forward(self, x):
        f = self._features(x)

        f_flat = torch.flatten(f, 1)

        if isinstance(self._num_classes, (list, set)):
            x = [fc(f_flat) for fc in self._fc]
            p = [self._activation(y) for y in x]
            return [{"prediction": p, "logits": x, "feature": f_flat} for p, x in zip(p, x)]

        x = self._fc(f_flat)
        p = self._activation(x)
        return {"prediction": p, "logits": x, "feature": f_flat}
