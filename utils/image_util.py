import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc("font", family='Microsoft YaHei')
# matplotlib.use('QtAgg')


def write_image(tensor, save_path=None, show_image=False, plot_kwargs=dict(), save_kwargs=dict()):
    """
    可视化一个二维矩阵
    :param tensor: 输入的二维矩阵
    :param save_path: 保存路径
    :param show_image: 显示图片
    :param plot_kwargs: 显示选项 plot_kwargs={'cmap': 'viridis'}
    :param save_kwargs: 保存选项 save_kwargs={'dpi': 1024}
    :return: 无
    """
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots()
    ax.imshow(tensor, **plot_kwargs)
    ax.axis('off')  # 关闭坐标轴
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if save_path is not None:
        fig.savefig(save_path, **save_kwargs)
    if show_image:
        plt.show()


def write_contour(tensor, save_path=None, show_image=False, plot_kwargs=dict(), save_kwargs=dict()):
    """
    可视化一个二维矩阵
    :param tensor: 输入的二维矩阵
    :param save_path: 保存路径
    :param show_image: 显示图片
    :param plot_kwargs: 显示选项 plot_kwargs={'cmap': 'viridis'}
    :param save_kwargs: 保存选项 save_kwargs={'dpi': 1024}
    :return: 无
    """
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots()
    ax.contourf(tensor, **plot_kwargs)
    ax.axis('off')  # 关闭坐标轴
    if save_path is not None:
        fig.savefig(save_path, **save_kwargs)
    if show_image:
        plt.show()


def write_hist(tensor, save_path=None, show_image=False, plot_kwargs=dict(), save_kwargs=dict()):
    """
    可视化一个一维数组
    :param tensor: 输入的一维数组
    :param save_path: 保存路径
    :param show_image: 显示图片
    :param plot_kwargs: 显示选项 plot_kwargs={'cmap': 'viridis'}
    :param save_kwargs: 保存选项 save_kwargs={'dpi': 1024}
    :return: 无
    """
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()
    ax.hist(tensor, **plot_kwargs)
    # xmin
    xmin, xmax = np.min(tensor), np.max(tensor)
    ax.set(xlim=(xmin, xmax))
    ax.set_xlabel('方差')
    ax.set_ylabel('数量')
    if save_path is not None:
        fig.savefig(save_path, **save_kwargs)
    if show_image:
        plt.show()


if __name__ == '__main__':
    import torch
    random_tensor = torch.rand(5, 5)
    write_image(random_tensor, save_path='./test.png')
    write_contour(random_tensor, save_path='./test.png')