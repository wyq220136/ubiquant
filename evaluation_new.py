from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 假设训练数据输出到一个data.csv中，输入列为inputs，标签列为label
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 4e-3

class state_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        df = pd.read_csv("data.csv")
        self.inputs = df["prompts"].tolist()
        self.labels = df["value"].tolist()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        
    def __getitem__(self, index):
        item = self.tokenizer(self.inputs[index], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        label = torch.tensor([self.labels[index]], dtype=torch.float)
        return {'input_ids': item['input_ids'].squeeze(0), 'attention_mask': item['attention_mask'].squeeze(0)}, label
    
    def __len__(self):
        return len(self.labels)


class evaluation_net(nn.Module):
    def __init__(self):
        super(evaluation_net, self).__init__()
        self.basemodel = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        self.classifier = nn.Linear(256, 1)
        self.to(device)
        
    def forward(self, x):
        x = self.basemodel(**x)
        # output = torch.flatten(x, start_dim=1)
        out_tmp = x.last_hidden_state[:, 0]
        out_tmp = self.pre_classifier(out_tmp)
        outrelu = F.relu(out_tmp)
        out_tmp = self.classifier(outrelu)
        return out_tmp



class ImprovedTrainer:
    def __init__(self, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        
        # 数据集划分
        full_dataset = state_Dataset()
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_set, self.val_set = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 数据加载器
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size)
        
        # 模型与优化器
        self.model = evaluation_net().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # 训练状态跟踪
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        epoch_loss = []
        
        for batch in progress_bar:
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device).float()

            self.optimizer.zero_grad()
            outputs = self.model({'input_ids': input_ids, 'attention_mask': attention_mask})
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss.append(loss.item())
            progress_bar.set_postfix({
                'train_loss': f"{np.mean(epoch_loss):.4f}",
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return np.mean(epoch_loss)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = batch
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device).float()
                
                outputs = self.model({'input_ids': input_ids, 'attention_mask': attention_mask})
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        for epoch in range(self.epochs):
            # 训练阶段
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            
            # 打印epoch摘要
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {self.best_val_loss:.4f}")
            print("-" * 50)

        # 训练结束保存最终模型
        torch.save(self.model.state_dict(), "final_model.pth")
        print("Training completed. Final model saved.")
        
        # 返回训练历史
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }

if __name__ == "__main__":
    trainer = ImprovedTrainer()
    trainer.train()