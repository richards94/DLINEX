import torch
import os


def logEpoch(logger, model, epoch, loss, accuracy):
    # 1. Log scalar values (scalar summary)
    info = {'loss': loss.item(), 'accuracy': accuracy.item()}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    # 3. Log training images (image summary)
    # info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

    # for tag, images in info.items():
        # logger.image_summary(tag, images, epoch)


def save_checkpoint(state,workdir,
                    filename='checkpoint.pth.tar'):

    os.makedirs(workdir, exist_ok=True) # 确保目录存在
    filepath = os.path.join(workdir,
                            filename)
    torch.save(state, filepath)

