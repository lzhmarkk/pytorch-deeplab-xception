def show(image, target, label, dataset, output=None):
    """
    show image and pred and target
    """
    from torchvision.utils import make_grid
    from dataloaders.utils import decode_seg_map_sequence
    import matplotlib.pyplot as plt
    import torch

    image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
    target = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                               dataset=dataset), 3, normalize=False, range=(0, 255))

    image = image.numpy().transpose((1, 2, 0))
    target = target.numpy().transpose((1, 2, 0))

    plt.title('image-{}'.format(label))
    plt.imshow(image)
    plt.show()
    plt.title('target-{}'.format(label))
    plt.imshow(target)
    plt.show()

    if output is not None:
        file = open("/home/lzhmark/Desktop/a.txt", 'a')
        file.write(str(output) + '\n')
        file.close()
        output = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                   dataset=dataset), 3, normalize=False, range=(0, 255))
        output = output.numpy().transpose((1, 2, 0))
        plt.title('prediction-{}'.format(label))
        plt.imshow(output)
        plt.show()
