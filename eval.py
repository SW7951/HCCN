import argparse
import copy
import math
import os
import warnings
from functools import reduce

import matplotlib
import torch
import torchvision.transforms.functional as F
from PIL import ImageDraw, ImageFont, Image
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
from dataset import WSIClassificationDataset
from presets import ClassificationPresetEval
from train import get_model_finetuning_the_convnet

DATASET_ROOT_PATH = '../../datasets/Classification'
DEFAULT_MODEL_PATH = 'data/model.pth'
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_WORKERS = 8
classes = ['出血', '坏死', '实质', '淋巴', '空泡', '空白', '间质']

warnings.filterwarnings("ignore")


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    test_data = WSIClassificationDataset(DATASET_ROOT_PATH, type='val', transform=ClassificationPresetEval(crop_size=224))
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=opt.batch_size, num_workers=opt.workers)

    x, y = next(iter(copy.deepcopy(test_dataloader)))
    x, y = x.to(opt.device), y.to(opt.device)
    return test_dataloader, x, y


def test(dataloader, model, loss_fn, opt, num_classes):
    """
    测试模型
    :param dataloader:
    :param model:
    :param loss_fn:
    :param opt:
    :return:
    """
    y_true = torch.zeros((len(dataloader.dataset)), dtype=torch.int64)
    y_pred = torch.zeros((len(dataloader.dataset), num_classes), dtype=torch.float32)
    num_batches = len(dataloader)
    loss, confmat = 0, utils.ConfusionMatrix(num_classes)
    model.eval()  # Sets the module in evaluation mode
    with torch.no_grad():  # Disabling gradient calculation
        with tqdm(dataloader, desc='test', total=len(dataloader)) as pbar:  # 进度条
            for i, (X, y) in enumerate(pbar):
                X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
                pred = model(X)  # 预测结果
                loss += loss_fn(pred, y).item()  # 计算损失
                confmat.update(y.flatten(), pred.argmax(1).flatten())  # 更新指标
                pbar.set_postfix({'Avg loss': f'{loss / num_batches:>8f}'})
                y_true[i * opt.batch_size: min(i * opt.batch_size + opt.batch_size, len(dataloader.dataset))] = y
                y_pred[i * opt.batch_size: min(i * opt.batch_size + opt.batch_size, len(dataloader.dataset))] = torch.softmax(pred, dim=-1)
    return confmat, y_true, y_pred


def show_classification_result(images, labels, image_size=None, text_color=None):
    """
    展示图片分类结果
    :param images: 图片 images
    :param labels: 标签 list
    :param image_size: 图片大小
    :param text_color: 文本颜色
    :return:
    """
    # 预处理图片
    mean = torch.tensor([0.723, 0.485, 0.608]).reshape((3, 1, 1))  # 预训练时标准化的均值
    std = torch.tensor([0.293, 0.377, 0.333]).reshape((3, 1, 1))  # 预训练时标准化的方差
    images = [torch.clip(image.cpu() * std + mean, 0.0, 1.0) for image in images]  # 对输入tensor进行处理
    labels = [str(label) for label in labels]  # 对输入tensor进行处理

    # 绘制每张图
    scale = 16  # 设置字体缩放大小
    image_size = image_size or images[0].shape[1:]  # 获取图片大小
    font = ImageFont.truetype(font='data/Microsoft YaHei.ttf', size=sum(image_size) // scale)  # 设置字体

    num_images = len(images)
    for i in range(num_images):
        # 转换为PIL图像
        image = F.to_pil_image(images[i])
        image = image.resize(image_size)  # 放大以更清楚显示
        # 绘制标题
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), labels[i], font=font, fill=text_color)
        # 转换为tensor
        images[i] = F.pil_to_tensor(image)

    # 生成网格图
    nrow = 8
    # nrow = int(math.ceil(math.sqrt(len(images))))
    result = make_grid(images, nrow=nrow)
    result = F.to_pil_image(result)
    result.save('data/result.png')
    result.show()


def plot_confusion_matrix(y_true, y_score, classes):
    """
    计算并绘制混淆矩阵
    :param y_true: 真实标签，形状为[N,]
    :param y_score: 样本属于每个类别的概率，形状为[N, num_classes]
    :param num_classes: 类别
    """
    # 将预测概率转换为预测类别
    y_pred = torch.argmax(y_score, dim=1)
    # 计算类别数
    num_classes = len(classes)
    fig, ax = plt.subplots()
    colors = matplotlib.colormaps.get_cmap('Blues')
    font_prop = FontProperties(fname='data/Microsoft YaHei.ttf')
    # 创建混淆矩阵
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int)
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    # 按行进行归一化（按样本的每个类别进行归一化）
    row_sums = confusion_matrix.sum(dim=1).reshape((-1, 1))
    confusion_matrix = confusion_matrix / row_sums
    # 绘制图片
    im = ax.imshow(confusion_matrix, cmap=colors)
    ax.set_title("Confusion matrix")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(torch.arange(num_classes))
    ax.set_xticklabels(classes, fontproperties=font_prop)
    ax.set_yticks(torch.arange(num_classes))
    ax.set_yticklabels(classes, fontproperties=font_prop)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig('data/result_confusion_matrix.png', dpi=1024)
    Image.open('data/result_confusion_matrix.png').show()
    return confusion_matrix


def plot_pr_curve(y_true, y_score, classes):
    """
    计算并绘制多分类PR曲线
    :param y_true: 真实标签，形状为[N,]
    :param y_score: 样本属于每个类别的概率，形状为[N, num_classes]
    :param classes: 类别
    """
    # 计算类别数
    num_classes = len(classes)
    pr_auc = torch.zeros(num_classes)
    # 绘制PR曲线
    fig, ax = plt.subplots()
    colors = matplotlib.colormaps.get_cmap('tab20')
    font_prop = FontProperties(fname='data/Microsoft YaHei.ttf')
    for i in range(num_classes):
        y_true_i = (y_true == i)
        y_score_i = y_score[:, i]
        # 计算TP, FP, FN的数量
        sorted_indices = torch.argsort(y_score_i, descending=True)
        tp = y_true_i[sorted_indices].cumsum(dim=0)  # 真正例-实际正例，预测正例
        fp = (1 - y_true_i[sorted_indices].int()).cumsum(dim=0)  # 假正例-实际负例，预测正例
        fn = y_true_i.sum() - tp  # 假反例-实际正例，预测负例
        # 计算精确度和召回率
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # 计算PR-AUC
        pr_auc[i] = torch.trapz(precision, recall)
        # 绘制PR曲线
        ax.plot(recall, precision, color=colors(i), label=f'Class {classes[i]} (PR-AUC = {pr_auc[i]:.5f})')
        # ax.fill_between(recall.cpu(), precision.cpu(), alpha=0.2, color=colors(i))

    # 添加x、y轴标签及标题
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_title('PR Curve')

    # 添加图例
    # labels = [classes[i] for i in range(num_classes)]
    # handles = [plt.Rectangle((0, 0), 1, 1, color=colors(i)) for i in range(num_classes)]
    ax.legend(loc='lower right', prop=font_prop)
    plt.savefig('data/result_PR_Curve.png', dpi=1024)
    Image.open('data/result_PR_Curve.png').show()
    return pr_auc


def plot_roc_curve(y_true, y_score, classes):
    """
    计算并绘制多分类ROC曲线
    :param y_true: 真实标签，形状为[N,]
    :param y_score: 样本属于每个类别的概率，形状为[N, num_classes]
    :param num_classes: 类别
    """
    # 计算类别数
    num_classes = len(classes)
    auc = torch.zeros(num_classes)
    # 绘制ROC曲线
    fig, ax = plt.subplots()
    colors = matplotlib.colormaps.get_cmap('tab20')
    font_prop = FontProperties(fname='data/Microsoft YaHei.ttf')
    for i in range(num_classes):
        y_true_i = (y_true == i)
        y_score_i = y_score[:, i]
        # 计算TPR, FPR的数量
        sorted_indices = torch.argsort(y_score_i, descending=True)
        tp = y_true_i[sorted_indices].cumsum(dim=0)  # 真正例-实际正例，预测正例
        fp = (1 - y_true_i[sorted_indices].int()).cumsum(dim=0)  # 假正例-实际负例，预测正例
        fn = y_true_i.sum() - tp  # 假反例-实际正例，预测负例
        tn = len(y_true_i) - y_true_i.sum() - fp  # 真反例-实际负例，预测负例
        # 计算TPR和FPR
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)  # tn = (len(y_true_i) - y_true_i.sum()) - fp
        # 计算AUC
        auc[i] = torch.trapz(tpr, fpr)
        # 绘制ROC曲线
        ax.plot(fpr, tpr, color=colors(i), label=f'Class {classes[i]} (AUC = {auc[i]:.5f})')

    # 添加x、y轴标签及标题
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_title('ROC Curve')

    # 添加图例
    ax.legend(loc='lower right', prop=font_prop)
    plt.savefig('data/result_ROC_Curve.png', dpi=1024)
    Image.open('data/result_ROC_Curve.png').show()

    plt.show()
    return auc


def visualize(model, x, y):
    """
    可视化模型效果
    :param model:
    :param x:
    :param y:
    """
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        pred = model(x)
        predict = [classes[i] for i in pred.argmax(dim=1)]  # 预测值
        actual = [classes[i] for i in y]  # 真实值

        labels = [f'{predict[i]}' if predict[i] == actual[i] else f'{predict[i]}({actual[i]})'
                  for i in range(len(predict))]
        show_classification_result(x, labels)


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='model data path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', default=DEFAULT_WORKERS, help='max dataloader workers')
    return parser.parse_args()


def main(opt):
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    dataloader, x, y = get_test_data(opt)
    # 模型
    num_classes = len(classes)
    model = get_model_finetuning_the_convnet(num_classes).to(opt.device)
    # 损失
    loss_fn = nn.CrossEntropyLoss()  # 损失函数
    # 评估
    print('Model with best gAcc:')
    model.load_state_dict(torch.load(opt.model_path))
    result, y_true, y_pred = test(dataloader, model, loss_fn, opt, num_classes)
    print(result)
    visualize(model, x, y)

    confmat = plot_confusion_matrix(y_true, y_pred, classes)
    pr_auc = plot_pr_curve(y_true, y_pred, classes)
    auc = plot_roc_curve(y_true, y_pred, classes)
    print('Confusion Matrix:')
    print(confmat)
    torch.set_printoptions(precision=5)
    print('PR-AUC:')
    print(pr_auc)
    print('AUC:')
    print(auc)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
