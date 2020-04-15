import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# prepare train and test splits

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# initialize model and train/test functions

net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

best_acc = 0.0

def pgd_attack(inputs, targets):
    epsilon = 8.0
    num_steps = 1
    step_size = 0.1

    perturb = torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon)
    perturb = perturb.to(device)

    x = inputs + perturb
    x = torch.clamp(x, 0, 255)
    x = x.to(device)
    x.requires_grad = True
    
    # PGD
    for i in range(num_steps):
        outputs = net(x)
        loss = criterion(outputs, targets)
        deriv = torch.autograd.grad(loss, x)

        x = torch.add(x, step_size * torch.sign(deriv[0]))
        x = torch.max(torch.min(x, inputs+epsilon), inputs-epsilon)
        x = torch.clamp(x, 0, 255)

    return x.detach() 
        
def robust_train(epoch):
    print('\nEpoch: %d' % epoch)
    start = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        x = pgd_attack(inputs, targets)
        
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    end = time.time()

    print()
    print("Time Elapsed for Training: " + str(end - start))
    print("Training Loss: " + str(train_loss))
    print()
    
def robust_test(epoch):
    global best_acc
    start = time.time()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        x = pgd_attack(inputs, targets)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    best_acc = max(best_acc, acc)
    end = time.time()

    print("Time Elapsed for Testing: " + str(end - start))
    print("Testing Loss: " + str(test_loss))
    print("Accuracy for Epoch " + str(epoch) + ": " + str(acc))
    print("Best Accuracy: " + str(best_acc))
    print()

for epoch in range(100):
    robust_train(epoch)
    robust_test(epoch)