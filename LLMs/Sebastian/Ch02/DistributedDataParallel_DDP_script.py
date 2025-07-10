import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost' # 1 Address of the main node
    os.environ['MASTER_PORT'] = '12355' # 2. Any free port on the machine

    init_process_group(
        backend='nccl', #3 nccl stands for NVIDIA Collective Communication Library.
        rank=rank, # #4 rank refers to the index of the GPU we want to use.
        world_size=world_size, #5 world_size is the number of GPUs to use.
    )

    torch.cuda.set_device(rank) # #6 Sets the current GPU device on which tensors will be allocated and operations will be performed

class ToyDataset(Dataset):
    def __init__(self,X,y):
        self.features = X
        self.labels = y

    def __getitem__(self,index): #1. Instructions for retrieving exactly one data record and the corresponding label

        one_x = self.features[index] # 1.
        one_y = self.labels[index] # 1.
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0] #2 Instructions for returning the total length of the dataset


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        ''' 1. Coding the number of inputs and outputs as variables
        allows us to reuse the same code for datasets with different 
        numbers of features and classes
        '''
        super().__init__()

        self.layers = torch.nn.Sequential(
            # first hidden layer
            torch.nn.Linear(num_inputs, 30), #2. The Linear layer takes the number of input and output nodes as arguments.
            torch.nn.ReLU(),              #3. Nonlinear activation functions are placed between the hidden layers.

            # second hidden layer
            torch.nn.Linear(30,20),   #4. The number of output nodes of one hidden layer has to match the number of inputs of the next layer.
            torch.nn.ReLU(),


            # the output layer
            torch.nn.Linear(20, num_outputs),

        )

    def forward (self, x):
        logits = self.layers(x)

        return logits  # 5. The outputs of the last layer are called logits.

def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below #7 Distributed-Sampler takes care of the shuffling now.
        pin_memory=True, #8 Enables faster memory transfer when training on GPU 
        drop_last=True, #9 Splits the dataset into distinct, non-overlapping subsets for each process (GPU)
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(train_ds)  # NEW #10 Distributed-Sampler takes care of the shuffling now.
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader

# def prepare_dataset():
#         # insert the dataset preparation code here
#         train_loader = DataLoader(
#             dataset = train_ds,
#             batch_size= 2, 
#             shuffle= False, #7 Distributed-Sampler takes care of the shuffling now.
#             pin_memory= True, #8 Enables faster memory transfer when training on GPU
#             drop_last= True, #9 Splits the dataset into distinct, non-overlapping subsets for each process (GPU)


#             sampler= DistributedSampler(train_ds)
#         )
#         return train_loader, test_loader
        


# NEW: wrapper
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # NEW: initialize process groups

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # NEW: wrap model with DDP
    # the core model is now accessible as model.module

    for epoch in range(num_epochs):
        # NEW: Set sampler to ensure each epoch has a different shuffle order
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(rank), labels.to(rank)  # New: use rank
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # NEW (not in the book):
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )
    ####################################################

    destroy_process_group()  # NEW: cleanly exit distributed mode


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    # This script may not work for GPUs > 2 due to the small dataset
    # Run `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py` if you have GPUs > 2
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)

    # NEW: spawn new processes
    # note that spawn will automatically pass the rank
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size spawns one process per GPU

