import random
import torch
import math

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round

def global_aggregate(fed_args, script_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None, stacking=False, heter=False, local_ranks=None):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None

    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
    
    elif fed_args.fed_alg == 'fedavgm':
        # Momentum-based FedAvg
        for key in global_dict.keys():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]

    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            # In paper 'adaptive federated optimization', momentum is not used
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
            global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

    elif fed_args.fed_alg == 'feddp':
        for key in global_dict.keys():
            global_dict[key] = sum([(local_dict_list[client][key] + gaussian_noise(local_dict_list[client][key].shape, fed_args, script_args, local_dict_list[client][key].device)) * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
    elif fed_args.fed_alg == 'flora':
        weights_array = torch.tensor([sample_num_list[client] / sample_this_round for client in clients_this_round]).to(torch.device('npu'))
        for k, client_id in enumerate(clients_this_round):
            single_weights = local_dict_list[client_id]
            x = 0
            if stacking:
                if k == 0:
                    weighted_single_weights = single_weights
                    for key in weighted_single_weights.keys():
                        if heter:
                            x += 1
                            if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                                weighted_single_weights[key] = weighted_single_weights[key] * weights_array[k]
                        else:
                            if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                                weighted_single_weights[key] = weighted_single_weights[key] * weights_array[k]
                else:
                    for key in weighted_single_weights.keys():
                        if heter:
                            x += 1
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                new = [weighted_single_weights[key], single_weights[key] * weights_array[k]]
                                weighted_single_weights[key] = torch.cat(new, dim=0)
                        else:
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                new = [weighted_single_weights[key], single_weights[key] * weights_array[k]]
                                weighted_single_weights[key] = torch.cat(new, dim=0)

                        if heter:
                            if single_weights[key].shape[1] == local_ranks[client_id]:
                                new = [weighted_single_weights[key], single_weights[key] * weights_array[k]]
                                weighted_single_weights[key] = torch.cat(new, dim=1)
                        else:
                            if single_weights[key].shape[1] == local_ranks[client_id]:
                                new = [weighted_single_weights[key], single_weights[key] * weights_array[k]]
                                weighted_single_weights[key] = torch.cat(new, dim=1)
            else:
                if k == 0:
                    weighted_single_weights = {key: single_weights[key] * weights_array[k] for key in single_weights.keys()}
                else:
                    weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * weights_array[k] for key in single_weights.keys()}
        
        global_dict = weighted_single_weights



    else:   # Normal dataset-size-based aggregation 
        for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    
    return global_dict, global_auxiliary

def gaussian_noise(data_shape, fed_args, script_args, device):
    if script_args.dp_sigma is None:
        delta_l = 2 * script_args.learning_rate * script_args.dp_max_grad_norm / (script_args.dataset_sample / fed_args.num_clients)
        # sigma = np.sqrt(2 * np.log(1.25 / script_args.dp_delta)) / script_args.dp_epsilon
        q = fed_args.sample_clients / fed_args.num_clients
        sigma = delta_l * math.sqrt(2*q*fed_args.num_rounds*math.log(1/script_args.dp_delta)) / script_args.dp_epsilon
    else:
        sigma = script_args.dp_sigma
    return torch.normal(0, sigma, data_shape).to(device)