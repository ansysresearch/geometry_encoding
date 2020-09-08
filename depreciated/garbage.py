
# # custom loss function
# def custom_loss_fn(y_pred, y_true):
#     l1_val = nn.L1Loss()(y_pred, y_true)
#     out_pred = (y_pred > 0).type(torch.float64)
#     #out_pred.requires_grad = True
#     out = (y_pred > 0).type(torch.float64)
#     ce_val = nn.BCELoss()(out_pred, out)
#     return l1_val, ce_val