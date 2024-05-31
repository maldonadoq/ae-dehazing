import numpy as np
import matplotlib.pyplot as plt


def plot_historial_pytorch(historial):
    x = np.arange(1, len(historial) + 1)
    y_loss = np.array([hist[0] for hist in historial])
    y_ssim = np.array([hist[1]['ssim'] for hist in historial])
    y_psnr = np.array([hist[1]['psnr'] for hist in historial])

    fig = plt.figure(figsize=(20, 4))
    rows, cols = 1, 3

    fig.add_subplot(rows, cols, 1)
    plt.plot(x, y_loss)
    plt.title('Loss evolution')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    fig.add_subplot(rows, cols, 2)
    plt.plot(x, y_ssim)
    plt.title('SSIM evolution')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    fig.add_subplot(rows, cols, 3)
    plt.plot(x, y_psnr)
    plt.title('PSNR evolution')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.show()


def plot_predict_pytorch(results, transform):
    i = np.random.randint(len(results))
    out_image, in_image, pr_image = results[i]
    j = np.random.randint(out_image.shape[0])

    rows = 1
    columns = 3

    out_image = transform(out_image[j])
    in_image = transform(in_image[j])
    pr_image = transform(pr_image[j])

    fig = plt.figure(figsize=(15, 7))

    fig.add_subplot(rows, columns, 1)
    plt.imshow(in_image)

    fig.add_subplot(rows, columns, 2)
    plt.imshow(pr_image)

    fig.add_subplot(rows, columns, 3)
    plt.imshow(out_image)

    plt.show()


def save_images_pytorch(results, transform, size, name, folder='../images/prediction/'):
    index = 1
    for _, _, pr_image_batch in results:
        for pr_image in pr_image_batch:
            out_name = folder + '{}_{}_{}.png'.format(size, index, name)
            transform(pr_image).save(out_name)
            index += 1


def get_information(historial, collection):
    _, mssim, mpsnr = np.mean(historial, axis=0)
    _, issim, ipsnr = np.argmax(historial, axis=0)

    print("Mean SSIM:", mssim)
    print("Mean PSNR:", mpsnr)
    print("Best SSIM:", collection.images[issim][0])
    print("Best PSNR:", collection.images[ipsnr][0])
