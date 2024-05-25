import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
    """
    跟踪一系列数值，通过一个窗口计算得到窗口均值
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        # 双端队列
        self.deque = deque(maxlen=window_size)
        # 总值
        self.total = 0.0
        # 计次
        self.count = 0
        # 格式
        self.fmt = fmt

    def update(self, value, n=1):
        """
        更新n个值
        @param value: 数值
        @param n: 数量
        @return: 无
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        进程间同步 Warning: does not synchronize the deque!
        @return: 无
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """ 中位数 """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """ 平均值 """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """ 全局平均值 """
        return self.total / self.count

    @property
    def max(self):
        """ 最大值 """
        return max(self.deque)

    @property
    def value(self):
        """ 最后一个值 """
        return self.deque[-1]

    def __str__(self):
        """ 字符串 """
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class ConfusionMatrix:
    """ 混淆矩阵 """
    def __init__(self, num_classes):
        # 种类数量
        self.num_classes = num_classes
        # 混淆矩阵 mat[i][j]表示标签为i预测为j的数量
        self.mat = None

    def update(self, target, pred):
        """
        根据标签值与预测值更新混淆矩阵
        @param target: 标签值 y.flatten()
        @param pred: 预测值 pred.argmax(1).flatten()
        @return:
        """
        # 创建混淆矩阵
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        # 推理模式下计算
        with torch.inference_mode():
            # 有效性矩阵 1代表有效 0代表无效
            k = (target >= 0) & (target < n)
            # 筛选出有效的值计算索引
            # 对于一个样本(target, pred)，其对应的值为值为target * n + pred
            inds = n * target[k].to(torch.int64) + pred[k]
            # 计数并转换，更新混淆矩阵
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        """
        重置混淆矩阵
        @return: 无
        """
        self.mat.zero_()

    def compute(self):
        """
        计算
        @return: 全局准确度, 准确度, 交并比
        """
        h = self.mat.float()
        # 类别准确度
        acc = torch.diag(h) / h.sum(1)
        # 全局准确度
        gAcc = torch.diag(h).sum() / h.sum()
        # 平均准确度
        mAcc = acc.mean()
        return acc, gAcc, mAcc

    def reduce_from_all_processes(self):
        """
        进程间reduce？
        @return: 无
        """
        reduce_across_processes(self.mat)

    def __str__(self):
        """ 字符串 """
        acc, gAcc, mAcc = self.compute()
        acc_str = ', '.join([f"{i:.2f}%" for i in (acc * 100).tolist()])
        return f"gAcc: {gAcc.item() * 100:.2f}%, aAcc: {mAcc.item() * 100:.2f}%\n" + f"Acc: {acc_str}\n"


class MetricLogger:
    """ 度量记录器 """
    def __init__(self, delimiter="\t"):
        # 计量器 本质为字典 默认值为一个SmoothValue
        self.meters = defaultdict(SmoothedValue)
        # 分隔符
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        更新度量标准
        @param kwargs: 参数 形式为name=meter
        @return: 无
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                raise TypeError(f"This method expects the value of the input arguments to be of type float or int, instead got {type(v)}")
            # 更新第k个度量
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """ 属性 """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """ 字符串 """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        进程间同步
        @return: 无
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        添加度量器
        @param name: 度量名称
        @param meter: 度量标准
        @return: 无
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        打印
        @param iterable: 可迭代对象
        @param print_freq: 打印频率
        @param header: 头部
        @return: 无
        """
        i = 0
        # 检查需不需要header
        if not header:
            header = ""
        # 起止时间
        start_time = time.time()
        end = time.time()
        # 迭代时间
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        # 格式 ':xd' -> 长度为x的整数，可迭代对象长度为78 -> :2d, 1478 -> :4d
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        # # 打印信息
        if torch.cuda.is_available():
            # 若cuda可用则添加内存信息
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}", "max mem: {memory:.0f}",]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        # MB单位
        MB = 1024.0 * 1024.0
        # 遍历可迭代对象
        for obj in iterable:
            # 更新时间
            data_time.update(time.time() - end)
            yield obj  # 按需生成值
            iter_time.update(time.time() - end)
            # 如果需要打印
            if i % print_freq == 0:
                # 剩余秒
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                # 剩余时间字符串
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB,))
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)))
            i += 1
            # 更新结束时间
            end = time.time()
        # 总时间
        total_time = time.time() - start_time
        # 总时间字符串
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


def cat_list(images, fill_value=0):
    """
    将一组图像拼为一个批次 batch
    @param images: 图像
    @param fill_value: 填充值
    @return: 一个批次的图像
    """
    # 图像中的最大尺寸
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # 一个批次的形状
    batch_shape = (len(images),) + max_size
    # 一个批次的图像填充默认值
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # 将原来图像填充回去
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    """
    处理数据加载器组装一个批次的逻辑
    @param batch: 批次
    @return:
    """
    images, targets = list(zip(*batch))
    # 图像填充0
    batched_imgs = cat_list(images, fill_value=0)
    # 标签填充255
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def mkdir(path):
    """
    创建文件夹
    @param path: 路径
    @return: 无
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    设置分布式计算，该函数禁用了非主进程的打印，替换了print函数
    This function disables printing when not in master process
    @param is_master: 是否为主进程
    @return: 无
    """
    import builtins as __builtin__
    # 保存print
    builtin_print = __builtin__.print
    # 定义新的print
    def print(*args, **kwargs):
        # 找出force选项，若不存在则设为False
        force = kwargs.pop("force", False)
        # 如果为主进程，或非主进程设置了强制输出(force=True)
        if is_master or force:
            builtin_print(*args, **kwargs)
    # 将print设置为自定义的print函数
    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    检查分布式训练环境是否可用和是否已经初始化
    @return: 分布式训练环境是否可用和是否已经初始化
    """
    # 检查 PyTorch 是否支持分布式训练
    if not dist.is_available():
        return False
    # 检查当前 PyTorch 进程是否已经初始化为分布式进程
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    获取当前分布式训练环境的进程数量（world size）。在分布式训练中，world size 表示运行在所有节点上的总进程数
    @return: 所有节点上的总进程数
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    获取当前进程在分布式训练环境中的排名（rank）。在分布式训练中，每个进程都有一个唯一的排名，通常从 0 开始。
    @return: 当前进程在分布式训练环境中的排名
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    检查当前进程是否是主进程。在分布式训练环境中，主进程通常负责一些特殊的任务，例如数据加载、模型保存等。
    @return: 当前进程排名若为0则为主进程
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    在分布式训练环境中只在主进程保存模型。在分布式训练中，只有主进程通常负责保存模型，以避免多个进程同时尝试写入相同的文件。
    @param args: 参数
    @param kwargs: 关键字参数
    @return:
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    初始化分布式训练模式，用于在分布式训练模式下初始化相关参数和环境。
    @param args: 参数
    @return: 无
    """
    # 检查环境变量
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # 使用 SLURM 环境
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    # 检查参数中是否已经存在 rank
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return
    # 初始化分布式环境
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def reduce_across_processes(val):
    """
    在分布式训练环境中将不同进程的值进行归约操作
    @param val: 值
    @return:
    """
    # 若非分布式环境
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)
    # 分布式环境
    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    # 最终结果
    return t
