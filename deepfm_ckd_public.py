from __future__ import print_function

import os
import logging
import argparse
import functools
import sys

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'predict'])
parser.add_argument('prod', choices=['hot', 'gamora', 'nebula','all'])
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
parser.add_argument('--text', action="store_true")
args = parser.parse_args()


import tensorflow as tf
from tensorflow.keras.backend import expand_dims,repeat_elements,sum


def print_debug_op(x, name):
    pt_op = tf.Print(x, [x,tf.shape(x), tf.reduce_max(x), tf.reduce_min(x)],
        message="Debug %s" % (name), summarize=100)
    with tf.control_dependencies([pt_op]):
        x = tf.identity(x)
    return x


def get_sess_context(emb,group,dim):
   group = tf.reshape(group,[-1])
   index_equal = tf.equal(tf.expand_dims(group,1), tf.expand_dims(group,0))
   index_equal_int = tf.cast(index_equal, tf.int64)
   batch = tf.shape(index_equal_int)[0]
   index_equal_int = tf.reshape(index_equal_int,[batch,batch])
   cnt = tf.count_nonzero(index_equal_int,1)
   max_cnt =  tf.reduce_max(cnt)
   idx_tmp = tf.reshape(tf.where(index_equal_int>0),[-1,2])
   idx = tf.slice(idx_tmp,[0,1],[-1,1])
   range_temp = tf.range(0,batch)
   range_idx = tf.reshape(range_temp,[-1,1])
   range_all_idx =tf.tile(range_idx,[1,max_cnt])
   ##################mask########################
   mask_matrix=tf.sequence_mask(cnt,max_cnt)
   final_sum_idx = tf.boolean_mask(range_all_idx,mask_matrix)
   emb_select = tf.nn.embedding_lookup(emb, idx)
   emb_select = tf.reshape(emb_select,[-1,dim])
   segment_emb = tf.segment_sum(emb_select,final_sum_idx)
   return segment_emb


def compute_confidense_approx(main_loss,bias_loss, main_p, bias_p):
    confidense_main = 1/(main_loss + 1e-8)
    confidense_bias = 1/(bias_loss + 1e-8)
    weight_main = confidense_main/ (confidense_main + confidense_bias)
    weight_bias = confidense_bias/ (confidense_main + confidense_bias)
    confidense_p = weight_main *main_p + weight_bias * bias_p   
    return confidense_p

def pair_hinge_sigmoid(y, y_pred,  group):
    y = tf.reshape(y, (-1, 1))
    group = tf.reshape(group, (-1, 1))
    margin = 0.03
    y_pred = tf.clip_by_value(y_pred, 1e-16, 1.0-1e-5)
    index_equal = tf.equal(tf.expand_dims(group,1), tf.expand_dims(group,0))
    y_v1 = tf.expand_dims(y, 1)
    y_v2 = tf.expand_dims(y, 0)
    label_greater = tf.greater(y_v1, y_v2)
    mask = tf.logical_and(label_greater, index_equal)
    diff = tf.expand_dims(y_pred,1) - tf.expand_dims(y_pred,0)
    return tf.reduce_sum(tf.boolean_mask(diff, mask))



def anchorwise_hinge_loss(labels, preds, weights, anchor_labels, anchor_preds, anchor_weights, reduction="sum", hinge=1, pair_cnt=None):
    """ 与 anchor 计算 loss.

    Args:
        labels (tensor[int64]): [batch_size, 1]
        preds (tensor[float32]): [batch_size, 1]
        weights (tensor[float32]): [batch_size, 1]
        anchor_labels (tensor[float32]): [batch_size, 10(page_size)]
        anchor_preds (tensor[float32]): [batch_size, 10]
        anchor_weights (tensor[float32]): [batch_size, 10] (作为 mask 使用，只能取值为0/1)
        reduction (str, optional): [description]. Defaults to "sum".
        hinge (int, optional): [description]. Defaults to 1.
    """
    unreduced_losses = tf.nn.relu(hinge - (labels - anchor_labels) * (preds - anchor_preds))
    anchor_weights = tf.cast(tf.not_equal(labels, anchor_labels), tf.float32) * anchor_weights
    unreduced_losses = weights * anchor_weights * unreduced_losses

    # mean loss for pairs in same group.
    if pair_cnt is not None:
        pair_cnt = tf.squeeze(pair_cnt, axis=1)
    else:
        pair_cnt = tf.reduce_sum(anchor_weights, axis=1)
    unreduced_losses = tf.div_no_nan(tf.reduce_sum(unreduced_losses, axis=1), pair_cnt)

    # Aggregate by batch.
    if reduction == "sum":
        return tf.reduce_sum(unreduced_losses)
    elif reduction == "mean":
        return tf.reduce_mean(unreduced_losses)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

def exp_and_sigmoid_act(max_value):
    def func(x):
        return tf.minimum(tf.math.exp(x), max_value), tf.math.sigmoid(x)
    return func

@tf.custom_gradient
def sigmoid(x):
    y = tf.nn.sigmoid(x)

    def grad(dy):
        return dy * y * (1.0 - y)
    return y, grad


def dense_layer(inputs, units, activation, name, weight_name, extra_inputs=[]):
    if not isinstance(inputs, list):
      inputs = [inputs]

    assert(len(inputs) > 0)
    assert(len(inputs) == 1 or len(extra_inputs) == 0)
    with tf.name_scope(name):
        i = inputs[0]
        weight=tf.get_variable(weight_name, (i.get_shape()[1], units))
        bias_weight = tf.get_variable(weight_name + "_bias", (units))
        o=tf.matmul(i,weight,name=name+"_mul")+bias_weight

        for idx, extra_i in enumerate(inputs[1:]):
            weight = tf.get_variable(weight_name + '_extra_' + str(idx), (extra_i.get_shape()[1], units))
            o = o + tf.matmul(extra_i, weight, name=name + '_extra_mul_' + str(idx))

        with tf.variable_scope(name):
            for i, extra_input in enumerate(extra_inputs):
                extra_kernel = tf.get_variable("kernel_extra_{}".format(i), (extra_input.get_shape()[1], units))
                o += tf.matmul(extra_input, extra_kernel)

        if activation is not None:
            return activation(o)
        else:
            return o

def simple_dense_network(inputs, units, name, weight_name_template, act=tf.nn.relu, extra_inputs=[]):
    output = inputs
    for i, unit in enumerate(units):
        # output = tf.layers.Dense(unit, act, name='dense_{}_{}'.format(name, i))(output)
        output = dense_layer(output, unit, act, name='dense_{}_{}'.format(name, i),
                                 weight_name=weight_name_template.format(i + 1),
                                 extra_inputs=extra_inputs)
        extra_inputs = []
    return output

def simple_dense_network_poso(inputs, units, name, weight_name_template, poso_input, poso_name, poso_weight_name_template, act=tf.nn.relu, extra_inputs=[]):
  output = inputs
  for i, unit in enumerate(units):
    output = dense_layer(output, unit, act, name='dense_{}_{}'.format(name, i), weight_name=weight_name_template.format(i + 1), extra_inputs=extra_inputs) * \
            dense_layer(poso_input, unit, act, name='dense_{}_{}'.format(poso_name, i), weight_name=poso_weight_name_template.format(i + 1))
  
  return output


def tf_transformer_component(query_input, action_list_input, name, col, nh=8, action_item_size=152, att_emb_size=64):
    Q = tf.get_variable(name + 'q_trans_matrix', (col, att_emb_size * nh))  # [emb, att_emb * hn]
    K = tf.get_variable(name + 'k_trans_matrix', (action_item_size, att_emb_size * nh))
    V = tf.get_variable(name + 'v_trans_matrix', (action_item_size, att_emb_size * nh))
    querys = tf.tensordot(query_input, Q, axes=(-1, 0))  # (batch_size,sq_q,att_embedding_size*head_num)
    keys = tf.tensordot(action_list_input, K, axes=(-1, 0))
    values = tf.tensordot(action_list_input, V, axes=(-1, 0)) # (batch_size,sq_v,att_embedding_size*head_num) 

    querys = tf.stack(tf.split(querys, nh, axis=2))  # (head_num,batch_size,field_sizeq,att_embedding_size)
    keys = tf.stack(tf.split(keys, nh, axis=2))      # (head_num,batch_size,field_sizek,att_embedding_size)
    values = tf.stack(tf.split(values, nh, axis=2))  # (head_num,batch_size,field_sizev,att_embedding_size)

    inner_product = tf.matmul(querys, keys, transpose_b=True) / 8.0 # (head_num,batch_size,field_sizeq,field_sizek)
    normalized_att_scores = tf.nn.softmax(inner_product)   #(head_num,batch_size,field_sizeq,field_sizek)
    result = tf.matmul(normalized_att_scores, values)  # (head_num,batch_size,field_sizeq,att_embedding_sizev)
    result = tf.transpose(result,  perm=[1, 2, 0, 3]) # (batch_size,field_sizeq,hn, att_embedding_sizev)
    mha_result = tf.reshape(result, (rown, nh * att_emb_size))
    return mha_result


transformer_component = tf_transformer_component

# embedding 
def parse_function(data_proto):
    feature_description = {
        'duration_ms': tf.io.FixedLenFeature([], tf.float32),
        'fvtr_75': tf.io.FixedLenFeature([], tf.float32),
        'fvtr_75_play_weight': tf.io.FixedLenFeature([], tf.float32),
        'sid': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(data_proto, feature_description)

tfrecord_file = './'
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(parse_function)
duration = dataset['duration_ms'] / 1000.0
llsid = dataset['sid']

user_id_emb = new_embedding(dataset['user_id'], dim=32)
user_gender_emb = new_embedding(dataset['user_gender'], dim=32)
user_age_segment_emb = new_embedding(dataset['user_age_segment'], dim=32)
photo_id_emb = new_embedding(dataset['photo_id'], dim=32)
duration_emb = new_embedding(duration, dim=32)
author_id_emb = new_embedding(dataset['author_id'], dim=32)
hetu_cluster_id_emb = new_embedding(dataset['hetu_cluster_id'], dim=32)
real_show_index_emb = new_embedding(dataset['real_show_index'], dim=32)

slot_ids = tf.concat([user_id_emb, user_gender_emb, user_age_segment_emb, photo_id_emb, duration_emb, author_id_emb, hetu_cluster_id_emb], axis=1)
bias_ids = tf.concat([photo_id_emb, duration_emb, real_show_index_emb])

dim = 96
t_pos_context_emb = get_sess_context(bias_ids,llsid,dim)

pos_all_fea_f = t_pos_context_emb
#####train&serve_feas#######

with tf.name_scope('train'):
    update_inputs = slot_ids
    task_input = update_inputs

    fm_first_order = tf.reduce_sum(tf.multiply(task_input,task_input),1)
    fm_first_order = tf.expand_dims(fm_first_order, -1)

    summed_features_emb = tf.reduce_sum(task_input,1)
    summed_features_emb_square = tf.square(summed_features_emb)
    squared_features_emb = tf.square(task_input)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb,1)
    fm_second_order = 0.5 * tf.subtract(summed_features_emb_square,squared_sum_features_emb)
    fm_second_order = tf.expand_dims(fm_second_order, -1)


    nn_output = simple_dense_network(task_input, [255, 255, 127], 'nn_layer',  'nn_layers{}_param',  extra_inputs=[])
    basic_tower  = dense_layer(tf.concat([nn_output, fm_first_order, fm_second_order], 1),127, None, 'basic_tower',  'basic_tower_param')
    basic_logit  = dense_layer(tf.concat([nn_output, fm_first_order, fm_second_order], 1),1, None,'basic_logit_layer',  'basic_logit_layer_param')

    bias_tower = simple_dense_network(pos_all_fea_f, [225,225,127], 'bias_tower',  'bias_tower{}_param')
    bias_logit = dense_layer(bias_tower, 1, None, 'bias_logit_layer', 'bias_logit_layer_param')

    s_fpr_logit = dense_layer(tf.concat([nn_output, fm_first_order, fm_second_order], 1),  1, None, 's_finish_play_top_layer_fpr',  's_finish_play_top_layer_fpr_param')


    all_t_fpr       = tf.nn.sigmoid(basic_logit)
    all_t_fpr       =  tf.clip_by_value(all_t_fpr,1e-16, 1.0-1e-5)
    all_t_fpr_bias_p =  tf.nn.sigmoid(bias_logit)
    all_t_fpr_bias_p = tf.clip_by_value(all_t_fpr_bias_p, 1e-16, 1.0-1e-5)

    all_fpr       = tf.nn.sigmoid(s_fpr_logit)
    all_fpr       =  tf.clip_by_value(all_fpr,1e-16, 1.0-1e-5)
   
    fvtr_75 = dataset["fvtr_75"]
    fvtr_75_play_weight = dataset["fvtr_75_play_weight"]

    bias_loss_weights = 0.5
    fpr_bias_loss     =  tf.losses.log_loss(fvtr_75,              all_t_fpr_bias_p,  fvtr_75_play_weight,   reduction = "weighted_sum")
    bias_loss         =  fpr_bias_loss 

    t_main_fpr_loss   =  tf.losses.log_loss(fvtr_75,              basic_logit,       fvtr_75_play_weight,     reduction="weighted_sum") 

    t_fpr_emb_cons    =  compute_confidense_approx(t_main_fpr_loss,fpr_bias_loss, basic_tower, bias_tower)   
    t_fpr_logit_cons  =  dense_layer(t_fpr_emb_cons,  1, None, 't_fpr_logit_cons',  't_fpr_logit_cons_param')
    all_t_fpr_cons    =  tf.nn.sigmoid(t_fpr_logit_cons)
    all_t_fpr_cons    =  tf.clip_by_value(all_t_fpr_cons, 1e-16, 1.0-1e-5)
 
   
    # q_names, preds, labels, weights, auc
    targets = [
        ('all_t_fpr_cons',     all_t_fpr_cons,       fvtr_75,               fvtr_75_play_weight,  'auc'),
        ('all_fpr',            all_fpr,              fvtr_75,               fvtr_75_play_weight,  'auc'),
        ("all_t_fpr",          all_t_fpr,            fvtr_75,               fvtr_75_play_weight,  'auc'), 
    ]
    bias_targets = [
        ('all_t_fpr_bias_p',   all_t_fpr_bias_p,     fvtr_75,               fvtr_75_play_weight,  'auc'),
    ]
    q_name, preds, labels, weights, auc = zip(*targets)
    bias_q_name, bias_preds, bias_labels, bias_weights, auc = zip(*targets)
    eval_targets = targets[:] + bias_targets[:]    
    soft_weight = 1
     
   # #######################
   
    ######################
   
    soft_target_fpr = tf.stop_gradient(all_t_fpr_cons)
    soft_loss_fpr = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=soft_target_fpr,
                    logits=s_fpr_logit
                )

    soft_loss =  tf.reduce_sum(soft_target_fpr) 

    loss =  bias_loss_weights * bias_loss  + tf.losses.log_loss(labels, preds, weights,  reduction="weighted_sum") + soft_loss  #for ssl loss
   # #######################

    optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
    opt = optimizer.minimize(loss)
