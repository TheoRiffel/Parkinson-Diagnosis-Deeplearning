import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import os
from tqdm import tqdm

N_EPOCHS = 200
PATIENCE = 20

def train_model(model, train_loader, val_loader, criterion, optimizer,
               num_epochs, patience, path2bestmodel, device):
    """Treina só no train_loader, salva o modelo de menor train-loss, sem early stopping."""
    os.makedirs(path2bestmodel, exist_ok=True)
    model.to(device).train()

    best_val_loss = float('inf')
    train_losses = val_losses = []
    train_accs = val_accs = []
    patience_counter = 0
    final_epoch = 0

    for epoch in tqdm(range(1, num_epochs+1), desc="Epoch"):

        # Treinamento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / len(train_loader)
        acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(acc)

        # Validação
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_running_loss += loss.item()
                preds = out.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_avg_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_avg_loss)
        val_accs.append(val_acc)

        # Early stopping
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{path2bestmodel}/best_model.pth")
            final_epoch = epoch
        else:
            patience_counter += 1

        print(f"[{epoch}/{num_epochs}] train-loss: {avg_loss:.4f}  train-acc: {acc:.4f} | val-loss: {val_avg_loss:.4f}  val-acc: {val_acc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'epoch_end': final_epoch
    }

def train(model, train_loader, val_loader, class_weights, device):
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_metrics = train_model(model, train_loader, val_loader, criterion, optimizer,
                                num_epochs=N_EPOCHS,
                                patience=PATIENCE,
                                path2bestmodel=f"weights/",
                                device=device)
    
    return train_metrics

def evaluate(model, test_loader, device):
    model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print(all_preds)

    return {
        'acc': accuracy_score(all_labels, all_preds),
        'f1':  f1_score(all_labels, all_preds, average='weighted'),
        'recall':  recall_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted')
    }