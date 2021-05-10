# coding: utf-8
from src.gru_train_and_evaluate import *
from src.gru_teacher_models import *
import time
import torch.optim
from src.expressions_transfer import *
import json
import torch
import copy
import random
from itertools import chain

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

torch.cuda.set_device(0)
batch_size = 50
embedding_size = 128
hidden_size = 512
n_epochs = 140
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 8
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'

def get_train_test_fold(ori_path,prefix,data,pairs,group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    #test_id = test_id[:1500]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num


data = load_raw_data("data/Math_23K.json")
group_data = read_json("data/Math_23K_processed.json")

data = load_raw_data("data/Math_23K.json")

pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)

#test_fold = train_fold[:1500]

best_acc_fold = []

pairs_tested = test_fold
#pairs_trained = valid_fold
pairs_trained = train_fold

#for fold_t in range(5):
#    if fold_t == fold:
#        pairs_tested += fold_pairs[fold_t]
#    else:
#        pairs_trained += fold_pairs[fold_t]

input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)

#print('train_pairs[0]')
#print(train_pairs[0])
#exit()
# Initialize models
teacher_encoder = TeacherEncoderRNN(input_size=output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                        n_layers=n_layers)
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
teacher_classifier = PreClassify(hidden_size=hidden_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate,weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
teacher_optimizer = torch.optim.Adam(chain(teacher_encoder.parameters(),teacher_classifier.parameters()), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=25, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=25, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=25, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=25, gamma=0.5)
teacher_scheduler = torch.optim.lr_scheduler.StepLR(teacher_optimizer, step_size=25, gamma=0.5)
# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    teacher_encoder.cuda()
    teacher_classifier.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

loss_teacher = 0
loss = 0

for epoch in range(n_epochs):
    if epoch < 25:
        threshold = 0.15
    elif epoch < 50:
        threshold = 0.15
    else:
        threshold = 0.15
    loss_total = 0
    op_list = [0,1,2,3,4]
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
        num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    fake_batches = copy.deepcopy(output_batches)
    start = time.time()
    for idx in range(len(input_lengths)):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], graph_batches[idx],teacher_encoder,teacher_classifier,epoch,if_teacher = True)
        for pos in range(len(num_size_batches[idx])):
            if num_size_batches[idx][pos] == 1:
                continue
            if output_batches[idx][pos][0] >= 7:
                continue
            num_list = list(range(7,7+num_size_batches[idx][pos]))
            one_true_record = output_batches[idx][pos]
            one_fake_record = copy.deepcopy(one_true_record)
            while(one_true_record == one_fake_record):
                for pos_word in range(output_lengths[idx][pos]):
                    p = random.random()
                    if p < threshold:
                        if one_true_record[pos_word] < 5:
                            one_fake_record[pos_word] = random.choice(op_list)
                        #else:
                        #    one_fake_record[pos_word] = random.choice(num_list)
            fake_batches[idx][pos] = one_fake_record
        loss_teacher = teacher_pretrain(
                    input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                    num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, output_lang, 
                    num_pos_batches[idx],fake_batches[idx],graph_batches[idx],teacher_encoder,teacher_classifier, teacher_optimizer, if_multi=True)
        loss_total += loss
    print('teacher_loss:',loss_teacher.item())
    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    teacher_scheduler.step()
    with torch.no_grad():
        if epoch % 10 == 0 or epoch > n_epochs - 80:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            for test_batch in test_pairs:
                #print(test_batch)
                batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
                test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                        merge, output_lang, test_batch[5], batch_graph, beam_size=beam_size)
                val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            print(equation_ac, value_ac, eval_total)
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            if value_ac > 784:
                torch.save(encoder.state_dict(), "model/encoder")
                torch.save(predict.state_dict(), "model/predict")
                torch.save(generate.state_dict(), "model/generate")
                torch.save(merge.state_dict(), "model/merge")
            if value_ac > 790:
                torch.save(encoder.state_dict(), "model/encoder_best")
                torch.save(predict.state_dict(), "model/encoder_best")
                torch.save(generate.state_dict(), "model/encoder_best")
                torch.save(merge.state_dict(), "model/encoder_best")
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))
