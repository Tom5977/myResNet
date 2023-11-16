import torch
from torch import nn
from torchvision import models, datasets
from torchvision.transforms import v2
from torch.cuda.amp import autocast as autocast, GradScaler
import time


# %%
class Timer:
    """记录多次运行时间"""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def sum(self):
        """返回时间总和"""
        return sum(self.times)


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# %%


# net = models.resnet34()
net = models.resnet50()
trans = nn.Sequential(v2.Resize(224),
                      v2.RandomCrop(224, padding=14),
                      v2.RandomHorizontalFlip(p=0.5),
                      # v2.RandomPerspective(distortion_scale=0.4, p=0.5),
                      v2.ToImage(),
                      v2.ToDtype(torch.float32, scale=True),
                      v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      )
train_data = datasets.CIFAR100(root='cifar100', train=True, transform=trans, download=True)
test_data = datasets.CIFAR100(root='cifar100', train=False, transform=nn.Sequential(v2.Resize(224), v2.ToImage(),
                                                                                    v2.ToDtype(torch.float32,
                                                                                               scale=True),
                                                                                    v2.Normalize([0.485, 0.456, 0.406],
                                                                                                 [0.229, 0.224,
                                                                                                  0.225])),
                              download=True)

batch_size = 128
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=6)
test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, num_workers=6)
# %%
log = open('printlog', 'w')
torch.cuda.empty_cache()
lr, num_epochs = 0.1, 200


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)

device = torch.device('cuda:0')
print('training on', device)
scaler = GradScaler()
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

loss = nn.CrossEntropyLoss()
timer, num_batches = Timer(), len(train_iter)
counter = 0
best_acc = 0.85
for epoch in range(num_epochs):
    counter += 1
    if counter / 5 == 1:
        counter = 0
        lr = lr * 0.5
    # 训练损失之和，训练准确率之和，样本数
    metric = Accumulator(3)
    net.train()
    for i, (X, y) in enumerate(train_iter):
        timer.start()
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        # 混合精度
        with autocast():
            y_hat = net(X)
            l = loss(y_hat, y)
        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
        timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]

    scheduler.step()

    test_acc = evaluate_accuracy_gpu(net, test_iter)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(net.state_dict(), 'resnet18.params')
        torch.save(best_acc, 'best_acc')
        print('better test_acc', file=log, flush=True)
    print(f'train loss: {train_l}, train acc: {train_acc}, test acc: {test_acc}', file=log, flush=True)
print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
      f'test acc {test_acc:.3f}')
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
      f'on {str(device)}')
print(f' total time {timer.sum()}')