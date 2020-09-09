from torch.nn import CrossEntropyLoss


def triple_loss(preds, labels):
    start_preds, end_preds, class_preds = preds
    start_labels, end_labels, class_labels = labels

    start_loss = CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    class_loss = CrossEntropyLoss(ignore_index=-1)(class_preds, class_labels)

    return start_loss + end_loss + class_loss