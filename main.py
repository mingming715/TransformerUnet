import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.unet_model import UNet
from data import trainloader, valloader
import torch.optim as optim
import torchvision.transforms as transforms
import PIL

net = UNet(3, 1)
net.cuda()
criterion = nn.L1Loss()  #
# criterion = nn.MSELoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001) #


def dice_score(pred, target, smooth=1e-5):
    # binary cross entropy loss

    pred = pred#torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    # dice coefficient
    dice = 2.0 * (intersection + smooth) / (union + smooth)

    return dice.sum()

best = 0
if __name__ == '__main__':
    for epoch in range(10):  # 데이터셋을 수차례 반복합니다.
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            
            print(outputs[0].shape)
            
            outputs_copy = outputs[0].clone()
            if epoch >= 5:
                for row in range(128):
                    for col in range(128):
                        if outputs_copy[0][row][col]<0.75:
                            outputs_copy[0][row][col]=0.
                        else:
                            outputs_copy[0][row][col]=1.

            #     tf2 = transforms.ToPILImage()
            #     ops2 = tf2(labels[0])

            #     ops2.show()
                tf = transforms.ToPILImage()
                ops = tf(outputs_copy)
                ops.show()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()

            print(running_loss/(i+1))
            break

        net.eval()
        total = 0

        for i, data in enumerate(valloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()


            # 순전파 + 역전파 + 최적화를 한 후
            with torch.no_grad():
                outputs = net(inputs)
            
            # if i==0:
            #     tf = transforms.ToPILImage()
            #     output = tf(labels[7])
            #     output2 = tf(outputs[7])
            #     output.show()
            #     output2.show()
            score = dice_score(outputs, labels)

            # 통계를 출력합니다.
            total += score.item()
        score = total / (i + 1)
        if score > best:
            best=score
            torch.save(net.state_dict(),'best_model.pt')

        print('score: ', total / (i + 1), 'best: ',best)

        net.train()

    
    print('Finished Training')
    # nvidia-smi