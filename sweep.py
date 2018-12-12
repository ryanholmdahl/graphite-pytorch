from train import read_data, train
from itertools import product

# dataset, num_train_examples, sample_train_randomly,  identity_features, num_bond_types,
#                    num_atom_types, num_max_nodes,
#                    model_type, lr, epochs, autoregressive, encode_features,
#                    gcn_batch_norm, gcn_hiddens, gcn_aggs, gcn_relus,
#                    graphite_relu, graphite_layers, z_dim, z_agg, dropout, num_gen_samples, num_gen_conditions,
#                    evals_to_stop, eval_only, load_path, use_pos_weight

# Data and preprocessing
# dataset = 'all_molecules.pkl'
# num_train_examples = 10000
# sample_train_randomly = True
# identity_features = True
# num_bond_types = 5
# num_atom_types = 4
# num_max_nodes = 9
#
# # Model hyperparameters
# model_type = 'multi_gcn_feedback'
# lr = 0.001
# epochs = 2000
# autoregressive = 0.5
# encode_features = False
# gcn_batch_norm = False
# gcn_hiddens = [32, 32]
# gcn_aggs = ['mean', 'mean']
# gcn_relus = [False, True]
# graphite_relu = True
# graphite_layers = 3
# z_dim = 32
# z_agg = 'mean'
# dropout = 0.
# use_norm = False
# use_pos_weight = True
# load_path = None
#
# # Evaluation
# num_gen_samples = 100
# num_gen_conditions = 10
# num_images_per_condition = 1
# final_num_gen_samples = 1000
# final_num_gen_conditions = 1000
# final_num_images_per_condition = 0
# evals_to_stop = 100
# eval_only = False

v_lrs = [0.001, 0.0003, 0.003]
v_gcn_layers = [1, 2]
v_autoregressives = [0.5]
v_gcn_relus = ['all']
v_gcn_batch_norms = [True]
v_graphite_layers = [2, 3]
v_z_dims = [32]
v_use_pos_weights = [False, True]


dataset = 'all_molecules.pkl'
num_train_examples = 150000
sample_train_randomly = True
identity_features = True
num_bond_types = 5
num_atom_types = 4
num_max_nodes = 9

model_type = 'multi_gcn_feedback'
epochs = 2000
encode_features = False
graphite_relu = True
z_agg = 'mean'
dropout = 0.
load_path = None

num_gen_samples = 100
num_gen_conditions = 10
num_images_per_condition = 1
final_num_gen_samples = 1000
final_num_gen_conditions = 1000
final_num_images_per_condition = 10
evals_to_stop = 100
eval_only = False

preprocessed_data, xa_mappings = read_data(dataset, num_train_examples, sample_train_randomly, identity_features,
                                           num_bond_types, num_atom_types, num_max_nodes)

for lr, gcn_layer, autoregressive, gcn_relu, gcn_batch_norm, graphite_layer, z_dim, use_pos_weight in \
    product(v_lrs, v_gcn_layers, v_autoregressives, v_gcn_relus, v_gcn_batch_norms, v_graphite_layers, v_z_dims, v_use_pos_weights):
    gcn_hiddens = []
    gcn_aggs = []
    gcn_relus = []
    for l in range(gcn_layer):
        gcn_hiddens.append(z_dim)
        gcn_aggs.append('mean')
        gcn_relus.append(True if gcn_relu == 'all' or (gcn_relu == 'last' and l == gcn_layer - 1) else False)

    train(dataset, num_train_examples, sample_train_randomly, identity_features, num_bond_types, num_atom_types,
          num_max_nodes, model_type, lr, epochs, autoregressive, encode_features, gcn_batch_norm, gcn_hiddens,
          gcn_aggs, gcn_relus, graphite_relu, graphite_layer, z_dim, z_agg, dropout, num_gen_samples,
          num_gen_conditions, evals_to_stop, eval_only, load_path, use_pos_weight, preprocessed_data, xa_mappings,
          num_images_per_condition, final_num_gen_samples, final_num_gen_conditions, final_num_images_per_condition)
