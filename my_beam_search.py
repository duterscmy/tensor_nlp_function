# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:04:49 2019

@author: Administrator
"""
import tensorflow as tf
from update_one_step import update_one_step
def myBeamSearch(batch_states,sequence_length,k,begin_id,end_id):
    '''
    给定初始状态和序列长度，集束搜索得到topk个分数最高的id序列以及它们的分数
    input:
        batch_states: (batch,dim) tensor
        sequence_length: int
    output:
        sequence_ids:(batch,k,sequence_length)
        sequence_score:(batch,k)
    *在这里还要完成一个mask功能，也就是说当beam里的一个序列是以end_id结尾时，不再对这个序列的分数进行更新。
    维护一个mask向量(batch,k)用来进行表示
    *使用tf.batch_gather而不是tf.gather的原因是每个样本需要gather的位置是不同的，gather作用于第一个维度，batch_gather作用于第二个维度
    *tf.nn.top_k()是tensorflow提供的函数，输入(batch,n),返回最大的k个数及他们的索引(batch,k)(batch,k)
    '''
    batch_size = batch_states.shape.as_list()[0]
    #初始化状态和输入
    states = tf.tile(tf.expand_dims(batch_states,axis=1),(1,k,1))
    inputs = tf.tile([[begin_id]],(batch_size,k))
    
    #初始化top k个id序列和他们的分数
    sequence_ids = tf.zeros(shape=(batch_size,k,0))
    sequence_score = tf.zeros(shape=(batch_size,k),dtype=tf.float32)
    
    mask = tf.ones((batch_size,k),dtype=tf.float32)
    
    for i in range(sequence_length):
        
        #将topk个状态和输入送给RNN，得到新的states和预测的概率分布
        new_states, now_score = update_one_step(states,inputs)#(batch,k,states),(batch,k,num_class)
        
        #根据当前已有序列的分数+每个序列的概率分布，得到k*num_class种结果的分数
        now_score = tf.multiply(now_score,tf.tile(tf.expand_dims(mask,axis=-1),(1,1,now_score.shape.as_list()[-1])))
        all_score = now_score + tf.expand_dims(sequence_score,axis=-1)
        
        #选出topk个高的分数以及它们的索引
        sequence_score, indexs = tf.nn.top_k(tf.reshape(all_score,shape=(batch_size,None)))
        
        #得到这topk个分数所属哪个beam，以及它们对应的输出类别即下一时刻的输入
        beam_ids = indexs//k
        inputs = indexs%k
        
        #更新topk个states,更新topk个输出序列
        states = tf.batch_gather(new_states,beam_ids)
        sequence_ids = tf.concat(tf.batch_gather(sequence_ids,beam_ids),tf.expand_dims(inputs,axis=-1))
        
        #根据end_id也就是当前的inputs来更新mask
        mask = tf.multiply(tf.cast(inputs!=end_id,dtype=tf.float32),tf.batch_gather(mask,beam_ids))
    return sequence_ids,sequence_score
        
        
    