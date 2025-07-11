from sklearn.metrics import accuracy_score, classification_report
from torchmetrics.classification import MulticlassCohenKappa
import torch

def eval(model, test_loader, device):
    print("\n--- Evaluation on Test Set ---")
    model.eval()
    all_preds = []
    all_labels = []

    QWK = MulticlassCohenKappa(num_classes=6).to(device)

    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    print(f"Final Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Final Test QWK: {QWK(torch.tensor(all_labels, device=device), torch.tensor(all_preds, device=device)):.4f}")
    print("Final Classification Report:\n", classification_report(all_labels, all_preds, zero_division=0))