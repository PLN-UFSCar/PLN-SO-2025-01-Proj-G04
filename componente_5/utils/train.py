import torch.nn as nn
import torch
from utils.eval import eval

def train(model, train_loader, test_loader, num_epochs, lr, weight_decay,  device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


    model.to(device)

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_embeddings.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += batch_labels.size(0)
            correct_predictions += (predicted == batch_labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions

        print(f"|-> Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

        if (epoch+1) % 10 == 0: 
            eval(model=model, test_loader=test_loader, device=device)

    print("--- Training Finished ---")
