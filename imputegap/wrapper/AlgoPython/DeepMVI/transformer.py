# ===============================================================================================================
# SOURCE: https://github.com/pbansal5/DeepMVI
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/abs/2103.01600
# ===============================================================================================================

from imputegap.wrapper.AlgoPython.DeepMVI.utils import *
from imputegap.wrapper.AlgoPython.DeepMVI.loader import *
from imputegap.wrapper.AlgoPython.DeepMVI.model import *

interval = 0

def train(model,train_loader,val_loader,device, max_epoch=1000, patience=2, lr=0.001, verbose=True, deep_verbose=False):
    best_state_dict = model.state_dict()
    best_loss = float('inf')

    lr = lr
    optim = torch.optim.Adam(model.parameters(),lr=lr)

    iteration = 0
    start_epoch = 0
    tolerance_epoch = 0
    patience = patience
    max_epoch = max_epoch
    train_error = 0
    for epoch in range(start_epoch,max_epoch):
        if verbose:
            print ("Starting Epoch : %d"%epoch)

        for inp_,mask,residuals,context_info in train_loader :

            #print(f"\nNATERQ__LOGS__ train_loader; {inp_.shape = }\n")
            #print(f"\nNATERQ__LOGS__ train_loader; {mask.shape = }\n")

            inp_ = inp_.to(device).requires_grad_(True)
            loss = model(inp_,mask.to(device),residuals.to(device),context_info)
            optim.zero_grad()
            loss['mae'].backward()
            optim.step()
            iteration += 1
            train_error += float(loss['mae'].detach().cpu())

            if (iteration % interval == 0):
                model.eval()
                loss_mre_num,count = 0,0
                with torch.no_grad():
                    for inp_,mask,residuals,context_info in val_loader :

                        loss = model.validate(inp_.to(device),mask.to(device), residuals.to(device),context_info)
                        loss_mre_num += (loss['loss_values']).sum()
                        count += len(loss['loss_values'])

                if (float(loss_mre_num)/count < 0.99*best_loss):
                    tolerance_epoch = 0
                    best_loss = float(loss_mre_num)/count
                elif (float(loss_mre_num)/count < best_loss):
                    best_state_dict = model.state_dict()
                    tolerance_epoch += 1
                else :
                    tolerance_epoch += 1

                if verbose:
                    print ('\tdone validation, Patience : ',tolerance_epoch)
                    print ('\tvalidation loss : ', float(loss_mre_num/count))
                    print ('\ttrain loss : ', float(train_error/interval))
                model.train()
                train_error = 0
                if tolerance_epoch >= patience:
                    if verbose:
                        print ('Early Stopping')
                    return best_state_dict

    return best_state_dict

def test(model, test_loader, val_feats, device, deep_verbose=False):
    output_matrix = copy.deepcopy(val_feats)

    if deep_verbose:
        print(f"\n\tNATERQ__LOGS__ test; {output_matrix.shape = }\n")
        print(f"\n\tNATERQ__LOGS__ test; {len(test_loader) = }\n")

    model.eval()

    if deep_verbose:
        torch.set_printoptions(threshold=float('inf'), linewidth=200)

    with torch.no_grad():
        for inp_, mask, residuals, context_info in test_loader :

            if deep_verbose:
                print(f"\n\n\t\tNATERQ__LOGS__ test_loader; {inp_.shape = }")
                print(f"\t\tNATERQ__LOGS__ test_loader; {mask.shape = }\n")

                print(f"\n\t\tNATERQ__LOGS__ test_loader; {context_info = }")
                # Print all values
                for i, t in enumerate(context_info):
                    print(f"\n\t\tTensor {i}:")
                    print(f"\t\t{t}")
                    print(f"\t\t{t.shape}")

                print(f"\t\tNATERQ__LOGS__ test_loader; {residuals = }\n")
                print(f"\t\tNATERQ__LOGS__ test_loader; {residuals.shape = }\n")

                print(f"\n\t\tNATERQ__LOGS__ test_loader; {inp_ = }\n")
                print(f"\t\tNATERQ__LOGS__ test_loader; {mask = }\n\n")

            loss = model.validate(inp_.to(device), mask.to(device), residuals.to(device), context_info, test=True)

            if deep_verbose:
                print(f"\n\n\t\tNATERQ__LOGS__ test_loader; {context_info[1][0] = }")
                print(f"\t\tNATERQ__LOGS__ test_loader; {context_info[1][0] = }\n")

                print(f"\n\t\tNATERQ__LOGS__ test_loader; {mask.shape[1] = }")
                print(f"\t\tNATERQ__LOGS__ test_loader; {residuals = }\n")

                print(f"\n\t\tNATERQ__LOGS__ test_loader; {context_info[0][0,0] = }\n")

            output_matrix[context_info[1][0]:context_info[1][0]+mask.shape[1],context_info[0][0,0]] = \
            np.where(mask.detach().cpu().numpy()[0],loss['values'].detach().cpu().numpy()[0],output_matrix[context_info[1][0]:context_info[1][0]+mask.shape[1],context_info[0][0,0]])

    model.train()

    if deep_verbose:
        print(f"\n\tNATERQ__LOGS__ test; {output_matrix.shape = }")
        print(f"\n\tNATERQ__LOGS__ test; {output_matrix = }\n")

    return output_matrix


def transformer_recovery(input_feats, max_epoch=1000, patience=2, lr=0.001, batch_size=16, seed=0, verbose=True, deep_verbose=False):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global interval
    interval = 1000

    mean = np.nanmean(input_feats, axis=0)
    std = np.nanstd(input_feats, axis=0)
    input_feats = (input_feats-mean)/std
    num_missing = 10*min(max(int(input_feats.shape[0]/100),1),500)
    train_feats, val_feats, val_points, test_points, block_size, kernel_size = make_validation(input_feats,num_missing=num_missing)

    if deep_verbose:
        print(f"\nNATERQ__LOGS__; {input_feats.shape = }")
        print(f"NATERQ__LOGS__; {train_feats.shape = }")
        print(f"NATERQ__LOGS__; {val_feats.shape = }")
        print(f"NATERQ__LOGS__; {val_points.shape = }")
        print(f"NATERQ__LOGS__; {val_points = }")
        print(f"NATERQ__LOGS__; {test_points.shape = }")
        print(f"NATERQ__LOGS__; {test_points = }")
        print(f"NATERQ__LOGS__; {block_size = }")
        print(f"NATERQ__LOGS__; {kernel_size = }\n")
        print(f"NATERQ__LOGS__; {num_missing = }\n")
        print(f"\n\nNATERQ__LOGS__; {val_feats = }")
        print(f"NATERQ__LOGS__; {train_feats = }\n")

    if (block_size > 100):
        kernel_size = 20

    time_context = min(int(input_feats.shape[0]/2),30*kernel_size)

    use_embed= (not is_blackout(input_feats))
    use_context=(block_size <= kernel_size)
    use_local = (block_size < kernel_size)

    if batch_size == -1:
        batch_size = min(input_feats.shape[1]*int(input_feats.shape[0]/time_context), 16)

    if input_feats.shape[0] < batch_size:
        batch_size = 2
        kernel_size = 1

    if verbose:
        print(f"\tblock_size is {block_size} and kernel_size is {kernel_size} with a time_context of {time_context}")
        print('\tfinal batch size :', batch_size)
        print('\tuse Kernel Regression :', use_embed)
        print('\tuse Context in Keys :', use_context)
        print('\tuse Local Attention :', use_local, "\n")

    train_set = myDataset(train_feats, use_local, time_context = time_context)
    val_set = myValDataset(val_feats, val_points,False, use_local, time_context = time_context)
    test_set = myValDataset(val_feats, test_points,True, use_local, time_context = time_context)

    if deep_verbose:
        print(f"{train_set.feats.shape =}")
        print(f"{val_set.feats.shape =}")
        print(f"{test_set.feats.shape =}")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True, collate_fn=my_collate)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, drop_last=False, shuffle=True, collate_fn=my_collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, drop_last=False, shuffle=True, collate_fn=my_collate)

    model = OurModel(sizes=[train_feats.shape[1]], kernel_size=kernel_size, block_size=block_size, nhead=2, time_len=train_feats.shape[0], use_embed=use_embed, use_context=use_context, use_local=use_local).to(device)
    model.std = torch.from_numpy(std).to(device)

    best_state_dict = train(model, train_loader, val_loader, device, max_epoch=max_epoch, patience=patience, lr=lr, verbose=verbose, deep_verbose=deep_verbose)
    model.load_state_dict(best_state_dict)

    matrix = test(model, test_loader, val_feats, device, deep_verbose=deep_verbose)

    matrix = (matrix*std)+mean


    return matrix
